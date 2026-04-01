# models.py
import torch
import torch.nn as nn

# using Qwen3.5-2B

class AudioAdapter(nn.Module):
    """
    projection layer: acts as translator between HuBERT and Qwen
    
    """
    def __init__(self, audio_dim=1024, llm_dim=2048):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(audio_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, x):
        return self.proj(x)
    
class MELDQwenModel(nn.Module):
    def __init__(self, qwen_model, audio_dim=1024):
        super().__init__()
        self.qwen = qwen_model
        self.adapter = AudioAdapter(audio_dim, llm_dim=4096)

    def forward(self, audio_features, input_ids):
        # Translate audio to Qwen's language
        audio_embeds = self.adapter(audio_features) # [Batch, Seq, 4096]
        
        # Get Qwen's internal word embeddings for the text
        text_embeds = self.qwen.get_input_embeddings()(input_ids) # [Batch, Seq, 4096]
        
        # Stitch them together
        combined = torch.cat([audio_embeds, text_embeds], dim=1)
        
        # Run Qwen on the hybrid sequence
        return self.qwen(inputs_embeds=combined)

def create_prompt(transcript, tokenizer):
    messages = [
        {"role": "system", "content": "You are an expert in Speech Emotion Recognition."},
        {"role": "user", "content": f"Listen to the audio and read the transcript: '{transcript}'. What is the emotion? Choose from: Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise."}
    ]
    # This turns the list of messages into one single string with special tokens
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)