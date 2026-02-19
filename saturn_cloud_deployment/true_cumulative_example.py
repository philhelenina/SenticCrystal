"""
진짜 Cumulative Context 예시
===========================

Pure Cumulative = 현재까지의 모든 이전 발화 사용!
"""

def show_true_cumulative():
    """진짜 cumulative context 보여주기"""
    
    print("🎯 진짜 CUMULATIVE CONTEXT")
    print("=" * 50)
    print("70줄 대화에서 각 위치별 context:")
    print()
    
    dialogue_length = 70
    baseline_k = 10
    
    # 몇 개 위치만 샘플로 보여주기
    sample_positions = [0, 1, 5, 10, 20, 35, 50, 69]
    
    print(f"{'Position':<8} {'Baseline K=10':<20} {'Pure Cumulative':<25} {'차이'}")
    print("-" * 65)
    
    for pos in sample_positions:
        # Baseline K=10
        baseline_start = max(0, pos - baseline_k + 1)
        baseline_context = list(range(baseline_start, pos + 1))
        baseline_size = len(baseline_context)
        
        # Pure Cumulative (처음부터 현재까지 모든 발화)
        cumulative_context = list(range(0, pos + 1))
        cumulative_size = len(cumulative_context)
        
        # 표시용으로 축약
        if baseline_size <= 5:
            baseline_str = str(baseline_context)
        else:
            baseline_str = f"[{baseline_context[0]}...{baseline_context[-1]}]({baseline_size}개)"
            
        if cumulative_size <= 5:
            cumulative_str = str(cumulative_context)
        else:
            cumulative_str = f"[0...{pos}]({cumulative_size}개)"
        
        diff = cumulative_size - baseline_size
        
        print(f"{pos:<8} {baseline_str:<20} {cumulative_str:<25} +{diff}")
    
    print()
    print("🔍 구체적 예시:")
    
    # 70번째 발화 (마지막)
    pos = 69
    baseline_context = list(range(60, 70))  # [60,61,62,63,64,65,66,67,68,69]
    cumulative_context = list(range(0, 70))  # [0,1,2,...,68,69]
    
    print(f"\n📍 위치 {pos} (마지막 발화):")
    print(f"  Baseline K=10    : [60,61,62,63,64,65,66,67,68,69] (10개)")
    print(f"  Pure Cumulative  : [0,1,2,3,...,66,67,68,69] (70개)")
    print(f"  차이             : +60개!")


def compare_strategies():
    """다양한 cumulative 전략 비교"""
    
    print("\n🧠 CUMULATIVE 전략 비교")
    print("=" * 50)
    
    dialogue_length = 70
    test_position = 69  # 마지막 발화
    
    print(f"70줄 대화의 마지막 발화(위치 {test_position}) 예측시:")
    print()
    
    strategies = {
        "Baseline K=10": list(range(60, 70)),
        "Fixed K=20": list(range(50, 70)), 
        "Pure Cumulative": list(range(0, 70)),
        "Conservative": list(range(35, 70)),  # 절반 정도
        "Recent Heavy": list(range(55, 70))   # 최근 15개
    }
    
    for name, context in strategies.items():
        size = len(context)
        if size <= 10:
            range_str = str(context)
        else:
            range_str = f"[{context[0]}...{context[-1]}]"
        
        print(f"  {name:<15}: {range_str} ({size}개)")
    
    print(f"\n💡 Pure Cumulative는 정말로 처음부터 끝까지 모든 발화를 context로 사용!")


def show_computational_impact():
    """계산 복잡도 영향"""
    
    print(f"\n⚡ 계산 복잡도 영향")
    print("=" * 30)
    
    dialogue_lengths = [20, 50, 100, 200]
    baseline_k = 10
    
    print(f"{'대화길이':<8} {'Baseline':<12} {'Cumulative':<12} {'비율'}")
    print("-" * 40)
    
    for length in dialogue_lengths:
        baseline_ops = baseline_k * length  # 각 위치에서 K개씩
        cumulative_ops = sum(range(1, length + 1))  # 1+2+3+...+length
        ratio = cumulative_ops / baseline_ops
        
        print(f"{length:<8} {baseline_ops:<12} {cumulative_ops:<12} {ratio:.1f}x")
    
    print(f"\n💰 Pure Cumulative는 계산량이 훨씬 많아집니다!")
    print(f"하지만 더 풍부한 context 정보를 활용할 수 있습니다.")


def main():
    print("🎭 진짜 CUMULATIVE CONTEXT 이해")
    print("=" * 40)
    print("Pure Cumulative = 현재까지 모든 이전 발화!")
    print()
    
    show_true_cumulative()
    compare_strategies()
    show_computational_impact()
    
    print(f"\n🎯 결론:")
    print(f"- Pure Cumulative: 위치 N에서 [0,1,2,...,N-1,N] 모두 사용")
    print(f"- Baseline K=10: 위치 N에서 [N-9,N-8,...,N-1,N] 만 사용") 
    print(f"- 장점: 전체 대화 맥락 파악 가능")
    print(f"- 단점: 계산량 증가, 메모리 사용량 증가")
    print(f"- 여전히 forward-only (미래 정보는 사용 안 함)")


if __name__ == "__main__":
    main()