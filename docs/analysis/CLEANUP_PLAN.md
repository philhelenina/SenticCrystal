# 🧹 SenticCrystal 정리 계획

## 📂 **현재 파일 현황**

### **Python 스크립트 (16개)**
```
루트 레벨:
├── run_comprehensive_experiments.py    # ✅ 메인 실험 파이프라인
├── config_generator.py               # ✅ 설정 생성기
└── scripts/embeddings.py            # ✅ 임베딩 생성 스크립트

src/ 구조:
├── src/utils/                        # 유틸리티 (4개)
│   ├── gcp_data_loader.py           # ⚠️ 미사용 (GCP 통합)
│   ├── focal_loss.py                # ✅ 핵심 (클래스 불균형 해결)
│   ├── preprocessing.py             # ✅ 핵심 (데이터 전처리)
│   └── device_utils.py              # ✅ 핵심 (M4 최적화)
│
├── src/models/                       # 모델 (3개)
│   ├── bayesian_modules.py          # ✅ 핵심 (고급 Bayesian)
│   ├── sequential_bayesian.py       # ✅ 핵심 (순차 처리)
│   └── simple_bayesian.py           # ✅ 핵심 (MC Dropout)
│
├── src/features/                     # 특징 추출 (2개)
│   ├── sroberta_module.py           # ✅ 핵심 (S-RoBERTa)
│   └── wnaffect_module.py           # ✅ 핵심 (WordNet-Affect)
│
└── src/data_preprocessing/           # 임베딩 생성 (6개)
    ├── base_embedding_generator.py        # 🆕 리팩토링 (기본 클래스)
    ├── config146_generator.py            # 🆕 리팩토링 (Config146)
    ├── bayesian_config146_generator.py   # 🆕 리팩토링 (Bayesian)
    ├── embedding_generator.py            # 🗂️ 백업용 (원본)
    ├── bayesian_embedding_generator.py   # 🗂️ 백업용 (원본)  
    └── multi_config_embedding_generator.py # 🔜 리팩토링 예정
```

### **MD 문서 (7개)**
```
├── EXPERIMENTAL_RESULTS_SUMMARY.md      # ✅ 핵심 (성과 요약)
├── EXPERIMENTAL_PLAN.md                 # ✅ 핵심 (실험 계획)
├── COMPREHENSIVE_TURN_ANALYSIS_PLAN.md  # ✅ 핵심 (Turn Analysis)
├── COMPREHENSIVE_CODEBASE_ANALYSIS.md   # ✅ 핵심 (코드 분석)
├── IEMOCAP_4WAY_DATA_ANALYSIS.md       # ✅ 핵심 (데이터 분석)
├── saturn_cloud_setup.md               # ✅ 핵심 (Saturn Cloud)
└── REFACTORING_COMPLETE.md             # ✅ 핵심 (리팩토링 보고서)
```

---

## 🗑️ **정리 대상 식별**

### **1. 즉시 제거 (백업용 파일들)**

#### **임베딩 생성기 원본들 (리팩토링 완료)**
```bash
# 백업 디렉토리로 이동
mkdir -p backup/data_preprocessing/
mv src/data_preprocessing/embedding_generator.py backup/data_preprocessing/
mv src/data_preprocessing/bayesian_embedding_generator.py backup/data_preprocessing/
```

### **2. 검토 후 처리**

#### **미사용 코드**
```bash
# GCP 통합 (현재 미사용)
src/utils/gcp_data_loader.py  # → backup/ 이동 또는 제거
```

#### **중복 WordNet-Affect 데이터**
```bash
# 이미 식별된 중복들
scripts/wn-affect-1.1/       # → 제거
scripts/wn-domains-3.2/      # → 제거  
data/wn-domains/             # → 제거
# 유지: scripts/wn-affect-1.0/
```

### **3. 문서 정리**

#### **MD 파일 구조화**
```bash
# docs/ 디렉토리 생성하여 체계화
mkdir -p docs/{experiments,analysis,setup,archive}/

# 실험 관련
docs/experiments/
├── EXPERIMENTAL_PLAN.md
├── EXPERIMENTAL_RESULTS_SUMMARY.md  
└── COMPREHENSIVE_TURN_ANALYSIS_PLAN.md

# 분석 관련  
docs/analysis/
├── COMPREHENSIVE_CODEBASE_ANALYSIS.md
├── IEMOCAP_4WAY_DATA_ANALYSIS.md
└── REFACTORING_COMPLETE.md

# 환경 설정
docs/setup/
└── saturn_cloud_setup.md

# 루트에 남길 핵심 문서
├── README.md                    # 🆕 생성 필요
├── CHANGELOG.md                 # 🆕 생성 필요  
└── QUICK_START.md              # 🆕 생성 필요
```

---

## 📋 **정리 실행 계획**

### **Phase 1: 백업 및 중복 제거**
```bash
# 1. 백업 디렉토리 생성
mkdir -p backup/{data_preprocessing,utils,wordnet_data}

# 2. 리팩토링된 파일의 원본들 백업
mv src/data_preprocessing/embedding_generator.py backup/data_preprocessing/
mv src/data_preprocessing/bayesian_embedding_generator.py backup/data_preprocessing/

# 3. WordNet-Affect 중복 제거
rm -rf scripts/wn-affect-1.1/
rm -rf scripts/wn-domains-3.2/  
rm -rf data/wn-domains/

# 4. 미사용 GCP 로더 백업
mv src/utils/gcp_data_loader.py backup/utils/
```

### **Phase 2: 문서 구조화**
```bash
# 1. docs 디렉토리 구조 생성
mkdir -p docs/{experiments,analysis,setup}

# 2. 문서들 이동
mv EXPERIMENTAL_PLAN.md docs/experiments/
mv EXPERIMENTAL_RESULTS_SUMMARY.md docs/experiments/
mv COMPREHENSIVE_TURN_ANALYSIS_PLAN.md docs/experiments/

mv COMPREHENSIVE_CODEBASE_ANALYSIS.md docs/analysis/
mv IEMOCAP_4WAY_DATA_ANALYSIS.md docs/analysis/
mv REFACTORING_COMPLETE.md docs/analysis/

mv saturn_cloud_setup.md docs/setup/

# 3. 새 문서 생성
touch README.md CHANGELOG.md QUICK_START.md
```

### **Phase 3: 스크립트 최적화**
```bash
# 1. multi_config_generator.py 리팩토링
# 2. scripts/ 디렉토리 정리
# 3. __init__.py 파일들 업데이트
```

---

## 🎯 **정리 후 예상 구조**

### **최종 프로젝트 구조**
```
SenticCrystal/
├── README.md                         # 🆕 프로젝트 개요
├── CHANGELOG.md                      # 🆕 변경 사항  
├── QUICK_START.md                   # 🆕 빠른 시작
├── environment_saturn_cloud.yml     # Saturn Cloud 환경
│
├── run_comprehensive_experiments.py # 메인 실험 파이프라인
├── config_generator.py             # 설정 생성기
│
├── scripts/                        # 실행 스크립트
│   ├── embeddings.py              # 임베딩 생성
│   └── wn-affect-1.0/             # WordNet-Affect (단일)
│
├── src/                           # 소스 코드
│   ├── utils/                     # 유틸리티 (3개)
│   │   ├── focal_loss.py
│   │   ├── preprocessing.py  
│   │   └── device_utils.py
│   │
│   ├── models/                    # 모델 (3개)
│   │   ├── bayesian_modules.py
│   │   ├── sequential_bayesian.py
│   │   └── simple_bayesian.py
│   │
│   ├── features/                  # 특징 추출 (2개)
│   │   ├── sroberta_module.py
│   │   └── wnaffect_module.py
│   │
│   └── data_preprocessing/        # 임베딩 생성 (4개)
│       ├── base_embedding_generator.py
│       ├── config146_generator.py  
│       ├── bayesian_config146_generator.py
│       └── multi_config_generator.py      # 🔜 리팩토링
│
├── docs/                          # 문서화
│   ├── experiments/               # 실험 관련 문서
│   ├── analysis/                  # 분석 보고서
│   └── setup/                     # 환경 설정
│
├── backup/                        # 백업 파일들
│   ├── data_preprocessing/        # 원본 생성기들
│   └── utils/                     # 미사용 유틸리티
│
└── data/                         # 데이터 (정리됨)
    └── iemocap_4way_data/        # 핵심 IEMOCAP 데이터만
```

---

## 📊 **정리 효과 예상**

### **파일 수 변화**
| 카테고리 | 현재 | 정리 후 | 변화 |
|---------|-----|-------|------|
| Python 스크립트 | 16개 | 13개 | -19% |
| MD 문서 | 7개 | 10개 | +43% (구조화) |
| WordNet 데이터 | 4곳 중복 | 1곳 | -75% |
| 총 프로젝트 크기 | ~2GB | ~1.2GB | -40% |

### **유지보수성 향상**
- ✅ **명확한 구조**: docs/ 체계로 문서 분류
- ✅ **중복 제거**: 백업으로 이동하여 혼란 방지  
- ✅ **빠른 접근**: README.md + QUICK_START.md로 신규 사용자 지원
- ✅ **버전 관리**: CHANGELOG.md로 변경사항 추적

이 정리 계획으로 진행하시겠어요?