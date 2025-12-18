"""
Quantile Cumulative Strategy 설계
================================

대화 위치의 quantile에 따라 사용할 context 비율을 결정하는 전략
"""

import math

def quantile_cumulative_strategy(position, dialogue_length, quantile_weights=None):
    """
    Quantile-based cumulative context strategy
    
    Args:
        position: 현재 위치 (0-based)
        dialogue_length: 전체 대화 길이
        quantile_weights: quantile별 가중치 딕셔너리
    """
    
    if quantile_weights is None:
        # 기본 quantile 가중치
        quantile_weights = {
            0.25: 0.3,   # 25% 지점에서 30% context 사용
            0.50: 0.6,   # 50% 지점에서 60% context 사용  
            0.75: 0.85,  # 75% 지점에서 85% context 사용
            1.00: 1.0    # 100% 지점에서 전체 context 사용
        }
    
    # 현재 위치의 상대적 위치 (0.0 ~ 1.0)
    rel_position = position / max(dialogue_length - 1, 1)
    
    # 해당하는 quantile 찾기
    for quantile in sorted(quantile_weights.keys()):
        if rel_position <= quantile:
            context_ratio = quantile_weights[quantile]
            break
    else:
        context_ratio = 1.0  # 기본값
    
    # 사용 가능한 총 context 개수 (현재 위치까지)
    available_context = position + 1
    
    # quantile 비율에 따른 실제 사용할 context 개수
    actual_context_size = math.ceil(available_context * context_ratio)
    
    # Forward-only: 최근 N개 사용
    start_idx = max(0, position - actual_context_size + 1)
    context_range = list(range(start_idx, position + 1))
    
    return {
        'rel_position': rel_position,
        'quantile': quantile,
        'context_ratio': context_ratio,
        'available_context': available_context,
        'actual_context_size': actual_context_size,
        'context_range': context_range,
        'start_idx': start_idx
    }


def compare_quantile_strategies():
    """다양한 quantile 전략 비교"""
    
    print("🎯 QUANTILE CUMULATIVE 전략 비교")
    print("=" * 80)
    
    dialogue_length = 70
    test_positions = [17, 35, 52, 69]  # 25%, 50%, 75%, 100% 지점
    
    # 3가지 quantile 전략
    strategies = {
        "Conservative": {0.25: 0.2, 0.50: 0.4, 0.75: 0.7, 1.00: 1.0},
        "Balanced": {0.25: 0.3, 0.50: 0.6, 0.75: 0.85, 1.00: 1.0},
        "Aggressive": {0.25: 0.5, 0.50: 0.8, 0.75: 0.95, 1.00: 1.0}
    }
    
    baseline_k = 10
    
    print(f"70줄 대화에서 각 전략별 context 사용량:")
    print()
    
    for pos in test_positions:
        print(f"📍 위치 {pos} ({pos/69*100:.0f}% 지점)")
        
        # Baseline
        baseline_size = min(baseline_k, pos + 1)
        baseline_start = max(0, pos - baseline_size + 1)
        print(f"  Baseline K=10: [{baseline_start}...{pos}] ({baseline_size}개)")
        
        # Pure Cumulative
        print(f"  Pure Cumulative: [0...{pos}] ({pos+1}개)")
        
        # Quantile strategies
        for name, weights in strategies.items():
            result = quantile_cumulative_strategy(pos, dialogue_length, weights)
            start_idx = result['start_idx']
            size = result['actual_context_size']
            ratio = result['context_ratio']
            
            print(f"  {name:12}: [{start_idx}...{pos}] ({size}개, {ratio:.0%})")
        
        print()


def show_detailed_quantile_example():
    """구체적인 quantile 계산 예시"""
    
    print("🔍 QUANTILE CUMULATIVE 상세 계산")
    print("=" * 50)
    
    dialogue_length = 70
    position = 52  # 75% 지점
    
    print(f"예시: {dialogue_length}줄 대화의 위치 {position}")
    print()
    
    # Balanced quantile 전략
    weights = {0.25: 0.3, 0.50: 0.6, 0.75: 0.85, 1.00: 1.0}
    
    result = quantile_cumulative_strategy(position, dialogue_length, weights)
    
    print(f"1️⃣ 상대적 위치 계산:")
    print(f"   rel_position = {position} / {dialogue_length-1} = {result['rel_position']:.3f}")
    print()
    
    print(f"2️⃣ Quantile 매칭:")
    print(f"   {result['rel_position']:.3f} <= 0.75 quantile")
    print(f"   선택된 context 비율: {result['context_ratio']:.0%}")
    print()
    
    print(f"3️⃣ Context 크기 계산:")
    print(f"   사용 가능한 context: [0...{position}] = {result['available_context']}개")
    print(f"   실제 사용할 context: {result['available_context']} × {result['context_ratio']:.0%} = {result['actual_context_size']}개")
    print()
    
    print(f"4️⃣ Forward-only 적용:")
    print(f"   최근 {result['actual_context_size']}개 사용: [{result['start_idx']}...{position}]")
    print()
    
    print(f"🆚 다른 전략과 비교:")
    print(f"   Baseline K=10: [43...52] (10개)")
    print(f"   Pure Cumulative: [0...52] (53개)")
    print(f"   Quantile 85%: [{result['start_idx']}...52] ({result['actual_context_size']}개)")


def visualize_quantile_growth():
    """Quantile 전략의 context 증가 패턴 시각화"""
    
    print("\n📈 QUANTILE CONTEXT 증가 패턴")
    print("=" * 60)
    
    dialogue_length = 20  # 시각화를 위해 짧게
    weights = {0.25: 0.3, 0.50: 0.6, 0.75: 0.85, 1.00: 1.0}
    baseline_k = 5
    
    print(f"20줄 대화에서 위치별 context 크기:")
    print()
    
    positions = list(range(0, dialogue_length, 2))  # 짝수 위치만
    
    print(f"{'Pos':<4} {'Rel':<6} {'Quantile':<8} {'Available':<9} {'Ratio':<6} {'Actual':<6} {'Baseline':<8}")
    print("-" * 55)
    
    for pos in positions:
        result = quantile_cumulative_strategy(pos, dialogue_length, weights)
        baseline_size = min(baseline_k, pos + 1)
        
        print(f"{pos:<4} {result['rel_position']:<6.2f} {result['quantile']:<8.2f} "
              f"{result['available_context']:<9} {result['context_ratio']:<6.0%} "
              f"{result['actual_context_size']:<6} {baseline_size:<8}")
    
    print()
    print(f"💡 Quantile 전략의 특징:")
    print(f"   - 초기: 적은 비율로 시작 (30%)")
    print(f"   - 중기: 점진적 증가 (60%)")
    print(f"   - 후기: 대부분 사용 (85%)")
    print(f"   - 마지막: 전체 사용 (100%)")


def compare_computational_cost():
    """계산 비용 비교"""
    
    print(f"\n💰 계산 비용 비교")
    print("=" * 40)
    
    dialogue_lengths = [50, 100, 200]
    weights = {0.25: 0.3, 0.50: 0.6, 0.75: 0.85, 1.00: 1.0}
    baseline_k = 10
    
    print(f"{'Length':<8} {'Baseline':<10} {'Quantile':<10} {'Pure':<10} {'Q/B ratio':<10}")
    print("-" * 50)
    
    for length in dialogue_lengths:
        baseline_total = baseline_k * length
        
        quantile_total = 0
        pure_total = 0
        
        for pos in range(length):
            # Quantile cumulative
            result = quantile_cumulative_strategy(pos, length, weights)
            quantile_total += result['actual_context_size']
            
            # Pure cumulative
            pure_total += (pos + 1)
        
        q_ratio = quantile_total / baseline_total
        
        print(f"{length:<8} {baseline_total:<10} {quantile_total:<10} {pure_total:<10} {q_ratio:<10.1f}x")
    
    print(f"\n💡 Quantile은 Pure Cumulative보다 효율적이면서도")
    print(f"   Baseline보다 풍부한 context를 제공합니다!")


def main():
    print("🎭 QUANTILE CUMULATIVE STRATEGY 설계")
    print("=" * 50)
    print("위치별 quantile에 따라 context 비율을 조정하는 전략")
    print()
    
    compare_quantile_strategies()
    show_detailed_quantile_example()
    visualize_quantile_growth()
    compare_computational_cost()
    
    print(f"\n🎯 Quantile Cumulative의 장점:")
    print(f"   ✅ Pure Cumulative보다 계산 효율적")
    print(f"   ✅ Baseline보다 풍부한 context")
    print(f"   ✅ 위치에 따른 적응적 조정")
    print(f"   ✅ 여전히 forward-only 유지")
    print(f"   ✅ 하이퍼파라미터로 튜닝 가능")


if __name__ == "__main__":
    main()