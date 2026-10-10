FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/tmp/huggingface PYTHONPATH=/app/RLAIF/rebuild
WORKDIR /app/RLAIF
COPY rebuild/cloud/requirements.txt /tmp/requirements.txt
RUN pip uninstall -y ninja && pip install --no-cache-dir -r /tmp/requirements.txt && pip check \
    && python -c "from trl import SFTTrainer, DPOTrainer, DataCollatorForCompletionOnlyLM; from peft import LoraConfig; from google import genai"
COPY airdialogue_sampled_5k.jsonl ./
COPY rebuild ./rebuild
RUN PYTHONPATH=/app/RLAIF/rebuild python -c "import rlaif.training.ppo_core, rlaif.training.ppo_qwen, rlaif.rewards.ppo_reward, rlaif.training.ppo_adaptive, rlaif.training.ppo_qwen_adaptive" \
    && PYTHONPATH=/app/RLAIF/rebuild python -m unittest discover -s rebuild/rlaif/tests -p 'test*.py'
ENTRYPOINT ["python", "rebuild/cloud/run.py"]
