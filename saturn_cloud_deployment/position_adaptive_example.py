"""
Position Adaptive 구체적 예시
============================

15개 발화 대화에서 Position Adaptive가 어떻게 작동하는지 보여줍니다.
"""

def position_adaptive_logic(position, dialogue_length):
    """Position Adaptive 로직"""
    
    # 상대적 위치 계산 (0.0 = 시작, 1.0 = 끝)
    rel_position = position / max(dialogue_length - 1, 1)
    
    if rel_position <= 0.1:      # 초기 10%
        max_k = 3
    elif rel_position <= 0.3:    # 초기 30%
        max_k = 8
    elif rel_position <= 0.7:    # 중간 70%
        max_k = 15
    else:                        # 후기 30%
        max_k = 20
    
    # 실제 사용 가능한 context 크기 (현재 위치까지만)
    actual_k = min(max_k, position + 1)
    
    return max_k, actual_k, rel_position


def show_position_adaptive_example():
    """15개 발화에서 Position Adaptive 예시"""
    
    dialogue_length = 15
    baseline_k = 5  # 비교용 baseline
    
    print("🎯 15개 발화 대화에서 Position Adaptive vs Baseline K=5")
    print("=" * 80)
    print(f"대화 길이: {dialogue_length}")
    print(f"발화 위치: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14")
    print()
    
    print(f"{'Position':<8} {'Rel_Pos':<8} {'Phase':<12} {'Max_K':<6} {'Actual_K':<9} {'Baseline_K':<11} {'Context_Range':<20}")
    print("-" * 80)
    
    for pos in range(dialogue_length):
        max_k, actual_k, rel_pos = position_adaptive_logic(pos, dialogue_length)
        
        # Phase 결정
        if rel_pos <= 0.1:
            phase = "초기(10%)"
        elif rel_pos <= 0.3:
            phase = "초기(30%)"
        elif rel_pos <= 0.7:
            phase = "중간(70%)"
        else:
            phase = "후기(30%)"
        
        # Baseline context
        baseline_actual_k = min(baseline_k, pos + 1)
        baseline_start = max(0, pos - baseline_actual_k + 1)
        
        # Position adaptive context
        adaptive_start = max(0, pos - actual_k + 1)
        
        # Context range 표시
        baseline_range = f"[{baseline_start}-{pos}]"
        adaptive_range = f"[{adaptive_start}-{pos}]"
        
        print(f"{pos:<8} {rel_pos:<8.2f} {phase:<12} {max_k:<6} {actual_k:<9} {baseline_actual_k:<11} {adaptive_range:<20}")
    
    print()


def show_detailed_examples():
    """구체적인 몇 개 위치에서 자세한 예시"""
    
    dialogue_length = 15
    baseline_k = 5
    
    print("🔍 구체적인 예시들:")
    print("=" * 60)
    
    example_positions = [0, 2, 4, 7, 10, 14]
    
    for pos in example_positions:
        max_k, actual_k, rel_pos = position_adaptive_logic(pos, dialogue_length)
        
        print(f"\n📍 위치 {pos} (상대적 위치: {rel_pos:.2f})")
        
        # Baseline
        baseline_actual_k = min(baseline_k, pos + 1)
        baseline_start = max(0, pos - baseline_actual_k + 1)
        baseline_context = list(range(baseline_start, pos + 1))
        
        # Position Adaptive
        adaptive_start = max(0, pos - actual_k + 1)
        adaptive_context = list(range(adaptive_start, pos + 1))
        
        print(f"  Baseline K=5     : {baseline_context} (크기: {len(baseline_context)})")
        print(f"  Position Adaptive: {adaptive_context} (크기: {len(adaptive_context)})")
        print(f"  차이             : {len(adaptive_context) - len(baseline_context):+d}개")


def show_growth_pattern():
    """Context 크기 증가 패턴 시각화"""
    
    dialogue_length = 15
    baseline_k = 5
    
    print(f"\n📈 Context 크기 증가 패턴:")
    print("=" * 50)
    
    baseline_sizes = []
    adaptive_sizes = []
    
    for pos in range(dialogue_length):
        # Baseline
        baseline_size = min(baseline_k, pos + 1)
        baseline_sizes.append(baseline_size)
        
        # Position Adaptive
        max_k, actual_k, rel_pos = position_adaptive_logic(pos, dialogue_length)
        adaptive_sizes.append(actual_k)
    
    print(f"위치:     {' '.join(f'{i:2d}' for i in range(dialogue_length))}")
    print(f"Baseline: {' '.join(f'{s:2d}' for s in baseline_sizes)}")
    print(f"Adaptive: {' '.join(f'{s:2d}' for s in adaptive_sizes)}")
    
    print(f"\n최종 비교:")
    print(f"  Baseline 최대: {max(baseline_sizes)}")
    print(f"  Adaptive 최대: {max(adaptive_sizes)}")
    print(f"  Adaptive 평균: {sum(adaptive_sizes)/len(adaptive_sizes):.1f}")
    print(f"  Baseline 평균: {sum(baseline_sizes)/len(baseline_sizes):.1f}")


def main():
    print("🧠 POSITION ADAPTIVE 구체적 예시")
    print("=" * 50)
    print("15개 발화 대화에서 어떻게 작동하는지 보겠습니다.")
    print()
    
    # 전체 테이블
    show_position_adaptive_example()
    
    # 구체적 예시들
    show_detailed_examples()
    
    # 성장 패턴
    show_growth_pattern()
    
    print(f"\n🎯 핵심:")
    print(f"- Position Adaptive는 대화 위치에 따라 context 크기를 동적으로 조정")
    print(f"- 초기에는 작게, 후반에는 크게")
    print(f"- 모든 경우에 forward-only (미래 정보 사용 안 함)")
    print(f"- Baseline은 항상 고정 크기")


if __name__ == "__main__":
    main()