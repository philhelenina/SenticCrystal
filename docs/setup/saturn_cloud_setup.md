# Saturn Cloud A100 Setup Guide for SenticCrystal

## 🚀 **Quick Setup**

### **1. Saturn Cloud 워크스페이스 생성**
```bash
# Saturn Cloud에서 새 워크스페이스 생성
- Resource: A100 SXM4-80GB (2 GPUs 권장)
- Instance Type: ml.p4d.2xlarge 또는 ml.p4d.4xlarge
- Storage: 500GB+ EBS 볼륨
- Environment: Custom (아래 설정 사용)
```

### **2. Environment 설정**
```bash
# 프로젝트를 Saturn Cloud로 업로드
git clone <your-senticcrystal-repo>
cd SenticCrystal

# Conda 환경 생성
conda env create -f environment_saturn_cloud.yml
conda activate senticcrystal-saturn
```

### **3. 데이터 준비**
```bash
# IEMOCAP 데이터를 Saturn Cloud storage로 업로드
# 또는 S3/GCS에서 다운로드 설정
```

## ⚡ **Performance 최적화 설정**

### **A100 듀얼 GPU 설정**
```python
# PyTorch 설정 확인
import torch
print(f"Available GPUs: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")

# Multi-GPU 활용
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
    print("Multi-GPU mode enabled")
```

### **메모리 최적화**
```bash
# 환경 변수 설정 (environment.yml에 포함됨)
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:2048"
export CUDA_LAUNCH_BLOCKING=0
```

### **배치 크기 권장사항**
```python
# A100 80GB × 2 = 160GB VRAM 기준
training_configs = {
    "embedding_generation": {
        "batch_size": 64,      # S-RoBERTa 처리
        "context_turns": 6,    # K-turn windows  
    },
    "model_training": {
        "batch_size": 128,     # MLP 분류기
        "max_epochs": 200,
        "early_stopping": 10
    },
    "bayesian_training": {
        "batch_size": 64,      # 메모리 더 많이 사용
        "mc_samples": 10,      # Monte Carlo 샘플
    }
}
```

## 🔬 **실험 실행 가이드**

### **전체 실험 파이프라인**
```bash
# 1. Config146 임베딩 생성 (예상: 30-45분)
python scripts/embeddings.py --config config146 --context_turns 0,2,4,6

# 2. 종합 실험 실행 (예상: 1.5-2시간)  
python run_comprehensive_experiments.py --gpu_ids 0,1

# 3. 결과 분석 및 시각화
python analyze_results.py --results_dir results/
```

### **분산 처리 활용**
```python
# Dask를 활용한 병렬 처리 (environment.yml에 포함)
import dask
from dask.distributed import Client

client = Client('scheduler-address')  # Saturn Cloud Dask 클러스터
```

## 📊 **예상 성능**

### **Saturn Cloud A100 vs MacBook M4**
| 작업 | MacBook M4 | Saturn A100 | 가속 비율 |
|-----|-----------|-------------|----------|
| Config146 임베딩 (K=6) | 2-3시간 | 30-45분 | 4-5x |
| MLP 훈련 | 15-20분 | 3-5분 | 4-5x |
| Bayesian 훈련 | 45-60분 | 8-12분 | 5-6x |
| 전체 파이프라인 | 6-8시간 | 1.5-2시간 | 4x |

### **비용 최적화**
```bash
# Auto-shutdown 설정 (유료 시간 절약)
# Saturn Cloud UI에서 설정:
- Idle timeout: 30분
- Auto-shutdown: 2시간 유휴시
- 실험 완료 후 수동 종료 권장
```

## 🛠️ **Troubleshooting**

### **CUDA 메모리 부족시**
```python
# 배치 크기 줄이기
batch_size = 32  # 64에서 32로
context_turns = 4  # 6에서 4로

# 그래디언트 누적 사용
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps
```

### **모델 로딩 실패시**
```python
# 캐시 클리어
import torch
torch.cuda.empty_cache()

# HuggingFace 캐시 재설정
export HF_HOME="/tmp/huggingface_new"
```

### **네트워크 연결 문제시**
```bash
# 모델 사전 다운로드
python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('all-MiniLM-L6-v2')
print('Model cached')
"
```

## 📈 **모니터링 및 로깅**

### **실험 추적**
```python
# W&B 설정 (environment.yml에 포함)
import wandb
wandb.init(project="senticcrystal-saturn", 
          config={
              "platform": "saturn_cloud_a100",
              "gpu_count": torch.cuda.device_count(),
              "batch_size": 128
          })
```

### **리소스 모니터링**
```bash
# GPU 사용량 모니터링
watch -n 1 nvidia-smi

# 메모리 사용량 확인
python -c "
import torch
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.memory_allocated(i)/1e9:.1f}GB / {torch.cuda.max_memory_allocated(i)/1e9:.1f}GB')
"
```

## 🎯 **Best Practices**

1. **실험 전 체크리스트**:
   - [ ] GPU 메모리 확인
   - [ ] 데이터 업로드 완료  
   - [ ] 환경 변수 설정
   - [ ] Auto-shutdown 활성화

2. **실행 중 모니터링**:
   - [ ] GPU 사용률 90%+ 유지
   - [ ] 메모리 leak 없음 확인
   - [ ] 로그 실시간 확인

3. **실험 후 정리**:
   - [ ] 결과 다운로드/백업
   - [ ] 인스턴스 수동 종료
   - [ ] 비용 확인

이 설정으로 Saturn Cloud에서 효율적으로 SenticCrystal 실험을 진행할 수 있습니다!