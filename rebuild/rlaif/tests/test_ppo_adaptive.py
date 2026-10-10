"""CPU checks for adaptive continuation provenance and atomic update rollback."""
import copy
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from rlaif.training.ppo_adaptive import UpdateRejected, transactional_update, validate_parent
from rlaif.training.ppo_qwen import capture_rng, forward_response
from rlaif.training.ppo_qwen_adaptive import update_policy
from rlaif.tests.test_ppo_core import TinyPolicy, tiny_args


class AdaptiveTests(unittest.TestCase):
    def setup_model(self):
        policy = TinyPolicy()
        head = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW([{"params": [policy.logits], "lr": 1.5e-6},
                                      {"params": list(head.parameters()), "lr": 2e-5}])
        args = tiny_args()
        args.ppo_epochs = 1
        args.max_update_attempts = 4
        args.retry_lr_decay = .5
        args.min_learning_rate = 1e-7
        args.gate_fraction = .8
        args.advantage_clip = 5
        return policy, head, optimizer, args

    def test_rejected_attempt_restores_policy_value_adam_and_rng(self):
        p, h, opt, args = self.setup_model()
        # Include preexisting optimizer moments: rollback must restore those too.
        (p.logits.sum() + h.weight.sum()).backward()
        opt.step()
        opt.zero_grad()
        before = copy.deepcopy(opt.state_dict())
        weights = p.logits.detach().clone()
        head = h.weight.detach().clone()
        draws, lrs, events = [], [], []
        rollouts = object()
        def updater(policy, value, optimizer, batch, config, device):
            self.assertIs(batch, rollouts)
            self.assertTrue(torch.equal(policy.logits, weights))
            self.assertTrue(torch.equal(value.weight, head))
            self.assertTrue(torch.equal(optimizer.state[p.logits]["exp_avg"], before["state"][0]["exp_avg"]))
            draws.append((random.random(), torch.rand(1).item()))
            lrs.append(optimizer.param_groups[0]["lr"])
            (policy.logits.sum() + value.weight.sum()).backward()
            optimizer.step()
            if len(draws) == 1:
                raise UpdateRejected("test measured gate")
            return {"epochs": []}
        result = transactional_update(p, h, opt, rollouts, args, "cpu", updater, events.append)
        self.assertEqual(draws[0], draws[1])
        self.assertEqual(lrs, [1.5e-6, .75e-6])
        self.assertEqual([e["status"] for e in events], ["reserved", "rolled_back", "reserved", "accepted"])
        self.assertEqual(opt.param_groups[0]["lr"], .75e-6)
        self.assertEqual(len(result["adaptive_attempts"]), 2)

    def test_exhaustion_restores_all_parameters_and_original_lr(self):
        p, h, opt, args = self.setup_model()
        args.max_update_attempts = 2
        original = p.logits.detach().clone()
        def reject(*unused):
            with torch.no_grad():
                p.logits.add_(100)
                h.weight.add_(100)
            raise UpdateRejected("bad KL")
        head = h.weight.detach().clone()
        with self.assertRaises(UpdateRejected):
            transactional_update(p, h, opt, [], args, "cpu", reject)
        self.assertTrue(torch.equal(p.logits, original))
        self.assertTrue(torch.equal(h.weight, head))
        self.assertFalse(opt.state)
        self.assertEqual(opt.param_groups[0]["lr"], 1.5e-6)

    def test_storage_failure_after_acceptance_rolls_back(self):
        p, h, opt, args = self.setup_model()
        original = p.logits.detach().clone()
        def updater(*unused):
            with torch.no_grad():
                p.logits.add_(1)
            return {}
        def broken_sync(evidence):
            if evidence["status"] == "accepted":
                raise OSError("durable attempt upload failed")
        with self.assertRaises(OSError):
            transactional_update(p, h, opt, [], args, "cpu", updater, broken_sync)
        self.assertTrue(torch.equal(p.logits, original))

    def test_durable_prior_attempts_bound_restarts_and_decay_lr(self):
        p, h, opt, args = self.setup_model()
        observed = []
        def updater(*unused):
            observed.append(opt.param_groups[0]["lr"])
            return {}
        transactional_update(p, h, opt, [], args, "cpu", updater, prior_attempts=2)
        self.assertEqual(observed, [1.5e-6 * .25])
        with self.assertRaises(UpdateRejected):
            transactional_update(p, h, opt, [], args, "cpu", updater, prior_attempts=4)
        self.assertEqual(len(observed), 1)

    def test_nonfinite_is_restored_and_not_retried(self):
        p, h, opt, args = self.setup_model()
        calls = []
        def updater(*unused):
            calls.append(1)
            with torch.no_grad():
                p.logits.fill_(float("nan"))
            raise FloatingPointError("nonfinite gradient")
        with self.assertRaises(FloatingPointError):
            transactional_update(p, h, opt, [], args, "cpu", updater)
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.isfinite(p.logits).all())

    def test_parent_provenance_and_cursor_required(self):
        identity = dict(model="m", source_sha256="data", split_seed=7,
                        sft_adapter_sha256="sft", hyperparameters={"seed": 9})
        meta = dict(identity=identity, state=dict(updates=19, next_scenario=19))
        splits = dict(source_sha256="data", seed=7)
        validate_parent(meta, splits, "sft", "m", 9, 19)
        for change in ({"sft_adapter_sha256": "other"}, {"source_sha256": "heldout"}):
            bad = copy.deepcopy(meta)
            bad["identity"].update(change)
            with self.assertRaises(ValueError):
                validate_parent(bad, splits, "sft", "m", 9, 19)
        meta["state"]["next_scenario"] = 18
        with self.assertRaises(ValueError):
            validate_parent(meta, splits, "sft", "m", 9, 19)

    def test_cloud_config_keeps_reference_separate_and_target_cumulative(self):
        from cloud.prepare_ppo_adaptive import config
        spec = config("p", "us-central1", "tag", "20261009", "adaptive-v1")["workerPoolSpecs"][0]["containerSpec"]
        args = spec["args"]
        value = lambda flag: args[args.index(flag) + 1]
        self.assertEqual(spec["command"][-1], "rlaif.training.ppo_qwen_adaptive")
        self.assertTrue(value("--adapter").endswith("20261008-sft/sft/adapter"))
        self.assertTrue(value("--initial-checkpoint").endswith("20261009-ppo-v3/checkpoint-0019"))
        self.assertEqual(value("--max-updates"), "50")
        self.assertEqual(value("--starting-update"), "19")
        self.assertEqual(value("--ppo-epochs"), "1")
        self.assertEqual(value("--max-rollouts"), "124")

    def test_real_update_uses_one_epoch_and_headroom_gates(self):
        p, h, opt, args = self.setup_model()
        with torch.no_grad():
            old, val = forward_response(p, h, [0], [1, 2], "cpu")
        rows = [dict(prompt_ids=[0], response_ids=[1, 2], reward=reward,
                     old_logprobs=old.clone(), old_values=val.clone(), reference_logprobs=old.clone())
                for reward in (.5, -.5)]
        result = transactional_update(p, h, opt, rows, args, "cpu", update_policy)
        self.assertEqual(len(result["epochs"]), 1)
        self.assertEqual(result["advantage_clip"], 5)
        # Force measured KL .045: below unchanged hard .05, above approach .04.
        from rlaif.training.ppo_core import ppo_loss as real_loss
        calls = []
        def fake_loss(*pos, **kw):
            loss, diagnostic = real_loss(*pos, **kw)
            calls.append(1)
            if len(calls) > len(rows):
                diagnostic["approx_kl_old_new"] = torch.tensor(.045)
            return loss, diagnostic
        with torch.no_grad():
            old, val = forward_response(p, h, [0], [1, 2], "cpu")
        for row in rows:
            row.update(old_logprobs=old.clone(), old_values=val.clone())
        with patch("rlaif.training.ppo_core.ppo_loss", side_effect=fake_loss):
            with self.assertRaisesRegex(UpdateRejected, "approaches"):
                update_policy(p, h, opt, rows, args, "cpu")


if __name__ == "__main__":
    unittest.main()
