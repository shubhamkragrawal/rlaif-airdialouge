"""CPU numerical/regression checks for the real PPO objective and checkpoint state."""
import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import torch
from rlaif.training.ppo_core import (generalized_advantage, masked_mean, ppo_loss, reference_kl,
                      token_logprobs, token_rewards, whiten_advantages)
from rlaif.training.ppo_qwen import (batch_tensors, forward_response, restore_rng, save_checkpoint,
                      update_policy)


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(3))
        self.value_features = torch.nn.Parameter(torch.tensor([[1.0]]), requires_grad=False)

    def forward(self, input_ids, **kwargs):
        n = input_ids.shape[1]
        logits = self.logits.view(1, 1, 3).expand(1, n, 3)
        hidden = self.value_features.view(1, 1, 1).expand(1, n, 1)
        return SimpleNamespace(logits=logits, hidden_states=(hidden,))

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True)
        torch.save(self.state_dict(), Path(path) / "adapter_model.pt")


class TinyTokenizer:
    def save_pretrained(self, path):
        (Path(path) / "tokenizer.json").write_text("{}")


def tiny_args():
    return SimpleNamespace(kl_coef=.1, max_reference_kl=1, gamma=1., gae_lambda=.95,
        ppo_epochs=2, clip_range=.2, value_clip=.2, value_coef=.5, max_grad_norm=.3,
        max_update_kl=.05, max_ratio=4)


class PPOObjectiveTests(unittest.TestCase):
    def test_real_reference_kl_matches_categorical_kl_in_expectation(self):
        p, q = torch.tensor([.8, .2]), torch.tensor([.5, .5])
        mask = torch.ones(2, dtype=torch.bool)
        signed, k3 = reference_kl(p.log(), q.log(), mask)
        exact = torch.distributions.kl_divergence(torch.distributions.Categorical(p),
                                                torch.distributions.Categorical(q))
        self.assertAlmostEqual(float((p * signed).sum()), float(exact), places=6)
        self.assertAlmostEqual(float((p * k3).sum()), float(exact), places=6)
        self.assertTrue((k3 >= 0).all())
        equal, equal_k3 = reference_kl(p.log(), p.log(), mask)
        self.assertTrue(torch.equal(equal, torch.zeros_like(p)))
        self.assertTrue(torch.equal(equal_k3, torch.zeros_like(p)))

    def test_reference_changes_reward_not_policy_entropy(self):
        old = torch.tensor([[-.2, -.6]])
        mask = torch.ones_like(old, dtype=torch.bool)
        zero = torch.tensor([0.])
        same, _, _ = token_rewards(zero, old, old, mask, .1)
        changed, signed, _ = token_rewards(zero, old, old - .5, mask, .1)
        self.assertTrue(torch.equal(same, torch.zeros_like(old)))
        self.assertTrue(torch.allclose(signed, torch.full_like(old, .5)))
        self.assertTrue(torch.allclose(changed, torch.full_like(old, -.05)))

    def test_terminal_reward_and_padding_mask(self):
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
        logp = torch.zeros((2, 3))
        rewards, _, _ = token_rewards(torch.tensor([1., -1.]), logp, logp, mask, .1)
        self.assertTrue(torch.equal(rewards, torch.tensor([[0., 1., 0.], [-1., 0., 0.]])))

    def test_gae_terminal_bootstrap_and_padding(self):
        mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
        rewards = torch.tensor([[0., 1., 999.]])
        values = torch.tensor([[.1, .2, 999.]])
        advantages, returns = generalized_advantage(rewards, values, mask, gamma=1, lam=1)
        self.assertTrue(torch.allclose(advantages, torch.tensor([[.9, .8, 0.]])))
        self.assertTrue(torch.allclose(returns, torch.tensor([[1., 1., 0.]])))
        normalized = whiten_advantages(advantages, mask)
        self.assertAlmostEqual(float(masked_mean(normalized, mask)), 0, places=5)
        self.assertEqual(float(normalized[0, 2]), 0)

    def test_policy_clipping_for_positive_and_negative_advantages(self):
        new = torch.tensor([[1.5, .5]], requires_grad=True).log()
        new.retain_grad()
        zeros = torch.zeros_like(new)
        mask = torch.ones_like(new, dtype=torch.bool)
        loss, stats = ppo_loss(new, zeros, zeros, zeros, torch.tensor([[1., -1.]]),
                               zeros, mask, value_coef=0)
        self.assertAlmostEqual(float(loss), -.2, places=6)
        self.assertEqual(float(stats["clip_fraction"]), 1)
        loss.backward()
        self.assertTrue(torch.equal(new.grad, torch.zeros_like(new)))

    def test_value_loss_is_clipped(self):
        new_values = torch.tensor([[3.]], requires_grad=True)
        zero = torch.zeros((1, 1))
        mask = torch.ones((1, 1), dtype=torch.bool)
        _, stats = ppo_loss(zero, new_values, zero, zero, zero, torch.tensor([[2.]]),
                            mask, value_clip=.2)
        self.assertAlmostEqual(float(stats["value_loss"]), 1.62, places=5)
        self.assertEqual(float(stats["value_clip_fraction"]), 1)

    def test_old_policy_and_values_are_detached(self):
        old = torch.tensor([[-1.]], requires_grad=True)
        old_values = torch.tensor([[0.]], requires_grad=True)
        new = torch.tensor([[-1.]], requires_grad=True)
        new_values = torch.tensor([[.1]], requires_grad=True)
        loss, stats = ppo_loss(new, new_values, old, old_values,
            torch.ones((1, 1)), torch.ones((1, 1)), torch.ones((1, 1), dtype=torch.bool))
        loss.backward()
        self.assertIsNone(old.grad)
        self.assertIsNone(old_values.grad)
        self.assertEqual(float(stats["ratio_mean"]), 1)
        self.assertEqual(float(stats["approx_kl_old_new"]), 0)

    def test_nonfinite_and_extreme_ratios_abort(self):
        zero, one, mask = torch.zeros((1, 1)), torch.ones((1, 1)), torch.ones((1, 1), dtype=torch.bool)
        for new in [torch.tensor([[float("nan")]]), torch.tensor([[21.]])]:
            with self.assertRaises(FloatingPointError):
                ppo_loss(new, zero, zero, zero, one, one, mask)

    def test_padding_cannot_poison_loss_or_gradients(self):
        new = torch.tensor([[-1., float("nan")]], requires_grad=True)
        values = torch.tensor([[0., float("nan")]], requires_grad=True)
        old = torch.tensor([[-1., float("nan")]])
        mask = torch.tensor([[True, False]])
        target = torch.tensor([[1., float("nan")]])
        loss, _ = ppo_loss(new, values, old, old + 1, target, target, mask)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(new.grad).all())
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertEqual(float(new.grad[0, 1]), 0)

    def test_prompt_tokens_are_excluded_by_shift(self):
        class PositionalPolicy(torch.nn.Module):
            def forward(self, input_ids, **kwargs):
                n = input_ids.shape[1]
                logits = torch.zeros((1, n, 4))
                for pos in range(n):
                    logits[0, pos, pos % 4] = 3
                hidden = torch.arange(n, dtype=torch.float32).view(1, n, 1)
                return SimpleNamespace(logits=logits, hidden_states=(hidden,))
        value = torch.nn.Linear(1, 1)
        value.weight.data.fill_(1)
        value.bias.data.zero_()
        logps, values = forward_response(PositionalPolicy(), value, [0, 0, 0], [2, 3], "cpu")
        expected = token_logprobs(torch.tensor([[0., 0., 3., 0.], [0., 0., 0., 3.]]), torch.tensor([2, 3]))
        self.assertTrue(torch.equal(logps, expected))
        self.assertTrue(torch.equal(values, torch.tensor([2., 3.])))

    def test_update_improves_rewarded_action_with_frozen_snapshots(self):
        policy, value = TinyPolicy(), torch.nn.Linear(1, 1)
        value.weight.data.zero_()
        value.bias.data.zero_()
        optimizer = torch.optim.AdamW(list(policy.parameters()) + list(value.parameters()), lr=.01)
        rollouts = []
        for action, reward in [(1, 1.), (2, -1.)]:
            logp, values = forward_response(policy, value, [0], [action], "cpu")
            rollouts.append(dict(prompt_ids=[0], response_ids=[action], reward=reward,
                old_logprobs=logp.detach().clone(), old_values=values.detach().clone(),
                reference_logprobs=logp.detach().clone()))
        snapshots = copy.deepcopy(rollouts)
        measures = update_policy(policy, value, optimizer, rollouts, tiny_args(), "cpu")
        self.assertGreater(float(policy.logits[1]), float(policy.logits[2]))
        self.assertEqual(len(measures["epochs"]), 2)
        for new, old in zip(rollouts, snapshots):
            self.assertTrue(torch.equal(new["old_logprobs"], old["old_logprobs"]))
            self.assertTrue(torch.equal(new["old_values"], old["old_values"]))
        for group in measures["epochs"]:
            self.assertLess(group["approx_kl_old_new"], .05)

    def test_checkpoint_restores_optimizer_value_and_rng(self):
        policy, value = TinyPolicy(), torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(list(policy.parameters()) + list(value.parameters()), lr=.01)
        (policy.logits.sum() + value.weight.sum()).backward()
        optimizer.step()
        random.seed(12)
        torch.manual_seed(12)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = save_checkpoint(Path(tmp), policy, value, optimizer, TinyTokenizer(),
                {"updates": 1, "next_scenario": 1}, {"source_sha256": "immutable"})
            expected_random, expected_torch = random.random(), torch.rand(3)
            loaded = torch.load(checkpoint / "training_state.pt", weights_only=False)
            restored_value = torch.nn.Linear(1, 1)
            restored_value.load_state_dict(loaded["value_head"])
            self.assertTrue(torch.equal(restored_value.weight, value.weight))
            self.assertEqual(loaded["optimizer"]["state"][0]["step"], 1)
            self.assertEqual(loaded["state"]["next_scenario"], 1)
            restore_rng(loaded["rng"])
            self.assertEqual(random.random(), expected_random)
            self.assertTrue(torch.equal(torch.rand(3), expected_torch))
            latest = json.loads((Path(tmp) / "latest_checkpoint.json").read_text())
            self.assertEqual(latest["path"], "checkpoint-0001")

    def test_pinned_qwen_lora_forward_update_and_frozen_sft_reference(self):
        # Tiny randomly initialized architecture, no pretrained download or API call.
        from transformers import Qwen2Config, Qwen2ForCausalLM, GenerationConfig
        from peft import LoraConfig, PeftModel, get_peft_model
        torch.manual_seed(15)
        base = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=32,
            intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=64, eos_token_id=2,
            pad_token_id=0, bos_token_id=1))
        reference_base = copy.deepcopy(base)
        policy = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM"))
        with tempfile.TemporaryDirectory() as tmp:
            policy.save_pretrained(tmp)
            reference = PeftModel.from_pretrained(reference_base, tmp, is_trainable=False)
            reference.requires_grad_(False)
            reference.eval()
            policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            policy.enable_input_require_grads()
            policy.config.use_cache = False
            value = torch.nn.Linear(32, 1)
            value.weight.data.zero_()
            value.bias.data.zero_()
            optimizer = torch.optim.AdamW([
                {"params": [p for p in policy.parameters() if p.requires_grad], "lr": 1e-4},
                {"params": value.parameters(), "lr": 1e-4}])
            frozen = {name: p.detach().clone() for name, p in reference.named_parameters()}
            train_before = {name: p.detach().clone() for name, p in policy.named_parameters() if p.requires_grad}
            samples = []
            policy.eval()
            generation = GenerationConfig(do_sample=True, temperature=1, top_p=1, top_k=0,
                eos_token_id=2, pad_token_id=0, max_new_tokens=4)
            for index, score in enumerate([1., -1., 1., -1.]):
                prompt = [1, 4, 5]
                with torch.no_grad():
                    response = policy.generate(torch.tensor([prompt]), generation_config=generation)[0, 3:].tolist()
                    old, old_value = forward_response(policy, value, prompt, response, "cpu")
                    ref, _ = forward_response(reference, None, prompt, response, "cpu")
                samples.append(dict(prompt_ids=prompt, response_ids=response, reward=score,
                    old_logprobs=old.detach().clone(), old_values=old_value.detach().clone(),
                    reference_logprobs=ref.detach().clone()))
            result = update_policy(policy, value, optimizer, samples, tiny_args(), "cpu")
            self.assertEqual(len(result["epochs"]), 2)
            self.assertTrue(any(not torch.equal(p, train_before[name]) for name, p in policy.named_parameters()
                                if p.requires_grad))
            self.assertTrue(all(torch.equal(p, frozen[name]) for name, p in reference.named_parameters()))
            self.assertTrue(all(p.grad is None for p in reference.parameters()))


if __name__ == "__main__":
    unittest.main()
