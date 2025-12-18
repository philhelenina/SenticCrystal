"""
Quantile 하이퍼파라미터 튜닝 설명
==============================

Quantile 비율들을 실험을 통해 최적화하는 방법
"""

def explain_quantile_hyperparameters():
    """Quantile 하이퍼파라미터 설명"""
    
    print("🎛️ QUANTILE 하이퍼파라미터란?")
    print("=" * 50)
    
    print("Quantile 전략에서 조정 가능한 파라미터들:")
    print()
    
    # 기본 설정
    base_config = {
        "0.25 quantile": 0.30,  # 25% 지점에서 30% context 사용
        "0.50 quantile": 0.60,  # 50% 지점에서 60% context 사용  
        "0.75 quantile": 0.85,  # 75% 지점에서 85% context 사용
        "1.00 quantile": 1.00   # 100% 지점에서 100% context 사용
    }
    
    print("📋 기본 설정 (Base Configuration):")
    for quantile, ratio in base_config.items():
        print(f"  {quantile}: {ratio:.0%} context 사용")
    
    print()
    print("🔧 이 비율들을 실험을 통해 찾아야 하는 하이퍼파라미터입니다!")


def show_different_hyperparameter_settings():
    """다양한 하이퍼파라미터 설정 예시"""
    
    print("\n🧪 다양한 하이퍼파라미터 설정 예시")
    print("=" * 60)
    
    configs = {
        "Conservative": {
            "설명": "초기엔 매우 적게, 후반에만 많이 사용",
            "0.25": 0.10,  # 10%
            "0.50": 0.30,  # 30%
            "0.75": 0.60,  # 60%
            "1.00": 1.00   # 100%
        },
        "Aggressive": {
            "설명": "초기부터 많이 사용",
            "0.25": 0.50,  # 50%
            "0.50": 0.80,  # 80%
            "0.75": 0.95,  # 95%
            "1.00": 1.00   # 100%
        },
        "Linear": {
            "설명": "선형적으로 증가",
            "0.25": 0.25,  # 25%
            "0.50": 0.50,  # 50%
            "0.75": 0.75,  # 75%
            "1.00": 1.00   # 100%
        },
        "Exponential": {
            "설명": "후반부에서 급격히 증가",
            "0.25": 0.06,  # 6% (0.25²)
            "0.50": 0.25,  # 25% (0.50²)
            "0.75": 0.56,  # 56% (0.75²)
            "1.00": 1.00   # 100%
        }
    }
    
    # 70줄 대화의 위치 35 (50% 지점)에서 비교
    dialogue_length = 70
    test_position = 35
    available_context = test_position + 1  # 36개
    
    print(f"예시: 70줄 대화의 위치 {test_position} (50% 지점)에서 각 설정별 context 사용량:")
    print(f"사용 가능한 context: {available_context}개")
    print()
    
    for name, config in configs.items():
        ratio_50 = config["0.50"]
        actual_size = int(available_context * ratio_50)
        start_idx = max(0, test_position - actual_size + 1)
        
        print(f"{name:12}: {config['설명']}")
        print(f"             50% 지점에서 {ratio_50:.0%} 사용 → [{start_idx}...{test_position}] ({actual_size}개)")
        print()


def show_hyperparameter_search_process():
    """하이퍼파라미터 탐색 과정"""
    
    print("🔍 하이퍼파라미터 탐색 과정")
    print("=" * 40)
    
    print("1️⃣ 탐색 공간 정의:")
    print("   - 0.25 quantile: [0.1, 0.2, 0.3, 0.4, 0.5]")
    print("   - 0.50 quantile: [0.4, 0.5, 0.6, 0.7, 0.8]")  
    print("   - 0.75 quantile: [0.7, 0.8, 0.85, 0.9, 0.95]")
    print("   - 1.00 quantile: [1.0] (고정)")
    print()
    
    print("2️⃣ 제약 조건:")
    print("   - 0.25 ≤ 0.50 ≤ 0.75 ≤ 1.00 (단조증가)")
    print("   - 각 비율은 0.0~1.0 사이")
    print()
    
    print("3️⃣ 평가 방법:")
    print("   - Validation set에서 accuracy, macro-F1 측정")
    print("   - 5-fold cross validation")
    print("   - 계산 비용도 고려 (training time)")
    print()
    
    print("4️⃣ 탐색 방법:")
    print("   - Grid Search: 모든 조합 시도")
    print("   - Random Search: 랜덤 샘플링")
    print("   - Bayesian Optimization: 효율적 탐색")


def simulate_hyperparameter_search():
    """하이퍼파라미터 탐색 시뮬레이션"""
    
    print("\n🎯 하이퍼파라미터 탐색 시뮬레이션")
    print("=" * 50)
    
    # 가상의 실험 결과
    experiments = [
        {
            "config": {"0.25": 0.1, "0.50": 0.4, "0.75": 0.7, "1.00": 1.0},
            "accuracy": 0.723,
            "macro_f1": 0.715,
            "training_time": "45분"
        },
        {
            "config": {"0.25": 0.3, "0.50": 0.6, "0.75": 0.85, "1.00": 1.0},
            "accuracy": 0.741,  # 최고 성능
            "macro_f1": 0.738,
            "training_time": "67분"
        },
        {
            "config": {"0.25": 0.5, "0.50": 0.8, "0.75": 0.95, "1.00": 1.0},
            "accuracy": 0.728,
            "macro_f1": 0.721,
            "training_time": "89분"
        },
        {
            "config": {"0.25": 0.2, "0.50": 0.5, "0.75": 0.8, "1.00": 1.0},
            "accuracy": 0.735,
            "macro_f1": 0.729,
            "training_time": "56분"
        }
    ]
    
    print("실험 결과:")
    print(f"{'Config':<25} {'Accuracy':<10} {'Macro-F1':<10} {'Time':<10}")
    print("-" * 65)
    
    best_accuracy = 0
    best_config = None
    
    for i, exp in enumerate(experiments, 1):
        config_str = f"({exp['config']['0.25']:.1f},{exp['config']['0.50']:.1f},{exp['config']['0.75']:.2f},1.0)"
        
        marker = "🏆" if exp['accuracy'] > best_accuracy else "  "
        if exp['accuracy'] > best_accuracy:
            best_accuracy = exp['accuracy']
            best_config = exp['config']
        
        print(f"{marker} Config{i} {config_str:<20} {exp['accuracy']:<10.3f} {exp['macro_f1']:<10.3f} {exp['training_time']:<10}")
    
    print()
    print(f"🏆 최적 설정:")
    print(f"   25% 지점: {best_config['0.25']:.0%} context")
    print(f"   50% 지점: {best_config['0.50']:.0%} context")
    print(f"   75% 지점: {best_config['0.75']:.0%} context")
    print(f"   성능: {best_accuracy:.1%} accuracy")


def show_practical_tuning_tips():
    """실용적인 튜닝 팁"""
    
    print("\n💡 실용적인 튜닝 팁")
    print("=" * 30)
    
    print("✅ 시작점:")
    print("   - 기본값: (0.3, 0.6, 0.85, 1.0)부터 시작")
    print("   - Linear: (0.25, 0.5, 0.75, 1.0)도 좋은 시작점")
    print()
    
    print("✅ 탐색 전략:")
    print("   - 1단계: 넓은 범위에서 coarse search")
    print("   - 2단계: 좋은 영역에서 fine search")
    print("   - 3단계: 최종 검증")
    print()
    
    print("✅ 주의사항:")
    print("   - 단조증가 제약 조건 지키기")
    print("   - Overfitting 방지: validation set 사용")
    print("   - 계산 비용과 성능의 trade-off 고려")
    print()
    
    print("✅ 도메인별 조정:")
    print("   - 짧은 대화: 더 aggressive한 설정")
    print("   - 긴 대화: 더 conservative한 설정")
    print("   - 감정 변화가 급한 데이터: 초기 비율 높이기")


def main():
    print("🎛️ QUANTILE 하이퍼파라미터 튜닝")
    print("=" * 40)
    print("Quantile 비율들을 최적화하는 방법")
    print()
    
    explain_quantile_hyperparameters()
    show_different_hyperparameter_settings()
    show_hyperparameter_search_process()
    simulate_hyperparameter_search()
    show_practical_tuning_tips()
    
    print(f"\n🎯 요약:")
    print(f"- Quantile 비율 = 실험으로 찾아야 하는 하이퍼파라미터")
    print(f"- 다양한 설정을 시도해서 최고 성능 찾기")
    print(f"- 단조증가 제약 조건 지키면서 탐색")
    print(f"- Validation 성능으로 최적 설정 선택")


if __name__ == "__main__":
    main()