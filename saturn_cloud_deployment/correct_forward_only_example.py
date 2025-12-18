"""
올바른 Forward-Only Context 설명
===============================

Position Adaptive도 forward-only입니다!
미래 정보는 절대 사용하지 않고, 현재 시점 t까지만 사용합니다.
"""

# 실제 대화 예시
dialogue_example = [
    {"pos": 0,  "speaker": "F", "text": "Excuse me.", "label": "neu"},
    {"pos": 1,  "speaker": "M", "text": "Do you have your forms?", "label": "-1"},
    {"pos": 2,  "speaker": "F", "text": "Yeah.", "label": "neu"},
    {"pos": 3,  "speaker": "M", "text": "Let me see them.", "label": "-1"},
    {"pos": 4,  "speaker": "F", "text": "Is there a problem?", "label": "neu"},
    {"pos": 5,  "speaker": "M", "text": "Who told you to get in this line?", "label": "-1"},
    {"pos": 6,  "speaker": "F", "text": "You did.", "label": "-1"},
    {"pos": 7,  "speaker": "F", "text": "You were standing at the beginning and you directed me.", "label": "-1"},
    {"pos": 8,  "speaker": "M", "text": "Okay. But I didn't tell you to get in this line if you are filling out this particular form.", "label": "-1"},
    {"pos": 9,  "speaker": "F", "text": "Well what's the problem? Let me change it.", "label": "neu"},
    {"pos": 10, "speaker": "M", "text": "This form is a Z.X.four.", "label": "-1"},
    {"pos": 11, "speaker": "M", "text": "You can't-- This is not the line for Z.X.four. If you're going to fill out the Z.X.four, you need to have a different form of ID.", "label": "-1"},
    {"pos": 12, "speaker": "F", "text": "What? I'm getting an ID. This is why I'm here. My wallet was stolen.", "label": "-1"},
    {"pos": 13, "speaker": "M", "text": "No. I need another set of ID to prove this is actually you.", "label": "-1"},
    {"pos": 14, "speaker": "F", "text": "How am I supposed to get an ID without an ID? How does a person get an ID in the first place?", "label": "-1"},
    {"pos": 15, "speaker": "M", "text": "I don't know. But I need an ID to pass this form along. I can't just send it along without an ID.", "label": "-1"},
    {"pos": 16, "speaker": "F", "text": "I'm here to get an ID.", "label": "-1"},
    {"pos": 17, "speaker": "M", "text": "No. I need another ID, a separate one.", "label": "-1"},
    {"pos": 18, "speaker": "F", "text": "Like what? Like a birth certificate?", "label": "-1"},
    {"pos": 19, "speaker": "M", "text": "A birth certificate, a passport...a student ID; didn't you go to school? Anything?", "label": "-1"},
    {"pos": 20, "speaker": "F", "text": "Who the hell has a birth certificate?", "label": "-1"},
    {"pos": 21, "speaker": "F", "text": "Yes but my wallet was stolen, I don't have anything. I don't have any credit cards, I don't have my ID. Don't you have things on file here?", "label": "-1"},
    {"pos": 22, "speaker": "M", "text": "Yeah. We keep it on file, but we need an ID to access that file.", "label": "-1"},
    {"pos": 23, "speaker": "F", "text": "That's out of control.", "label": "ang"},
]

def show_forward_only_context(target_position, context_size):
    """Forward-only context: [t-k+1, t-k+2, ..., t-1, t]"""
    
    print(f"🎯 위치 {target_position}에서 context_size={context_size}")
    print(f"Forward-only: [t-{context_size-1}, ..., t-1, t]")
    
    # Forward-only context 계산
    start_idx = max(0, target_position - context_size + 1)
    end_idx = target_position + 1  # target_position까지 포함
    
    context = dialogue_example[start_idx:end_idx]
    
    print(f"실제 사용 범위: pos {start_idx} ~ {target_position}")
    print(f"Context 크기: {len(context)}")
    print()
    
    for i, utt in enumerate(context):
        marker = "🎯" if utt["pos"] == target_position else "📝"
        label_info = f"[{utt['label']}]" if utt["label"] != "-1" else "[unlabeled]"
        print(f"  {marker} pos={utt['pos']:2d} {utt['speaker']}: {utt['text'][:40]}... {label_info}")
    
    print(f"\n레이블 있는 발화: {sum(1 for utt in context if utt['label'] != '-1')}/{len(context)}")
    return context


def compare_strategies_correctly():
    """올바른 Forward-Only 비교"""
    
    print("🔍 FORWARD-ONLY CONTEXT 비교 (올바른 버전)")
    print("=" * 80)
    print("⚠️  중요: 둘 다 미래 정보는 절대 사용하지 않습니다!")
    print()
    
    target_positions = [9, 16, 23]  # 초기, 중기, 후기
    
    for target_pos in target_positions:
        dialogue_length = len(dialogue_example)
        rel_position = target_pos / (dialogue_length - 1)
        
        print(f"\n📍 위치 {target_pos} (상대적 위치: {rel_position:.2f})")
        print(f"현재 발화: '{dialogue_example[target_pos]['text'][:30]}...' [{dialogue_example[target_pos]['label']}]")
        print("-" * 60)
        
        # 1. Baseline (고정 K=10)
        print("1️⃣ BASELINE (고정 K=10):")
        baseline_context_size = min(10, target_pos + 1)
        baseline_context = show_forward_only_context(target_pos, baseline_context_size)
        
        print("\n" + "-" * 40)
        
        # 2. Position Adaptive
        print("2️⃣ POSITION ADAPTIVE:")
        
        # Position adaptive 로직 (최대 context 크기 결정)
        if rel_position <= 0.1:      # 초기 10%
            max_context = 3
        elif rel_position <= 0.3:    # 초기 30%
            max_context = 8  
        elif rel_position <= 0.7:    # 중간 70%
            max_context = 15
        else:                        # 후기 30%
            max_context = 20
        
        adaptive_context_size = min(max_context, target_pos + 1)  # 현재 위치까지만!
        
        print(f"위치 기반 최대 context: {max_context}")
        print(f"실제 사용 가능: {target_pos + 1} (현재까지)")
        print(f"최종 context 크기: {adaptive_context_size}")
        
        adaptive_context = show_forward_only_context(target_pos, adaptive_context_size)
        
        # 비교
        print(f"\n📊 비교:")
        print(f"  Baseline    : {len(baseline_context)}개 발화")
        print(f"  Adaptive    : {len(adaptive_context)}개 발화")
        print(f"  차이        : {len(adaptive_context) - len(baseline_context):+d}개")
        
        print("=" * 80)


def show_cumulative_vs_forward_only():
    """Cumulative과 Forward-only의 차이점 명확화"""
    
    print("\n🤔 용어 정리: CUMULATIVE vs FORWARD-ONLY")
    print("=" * 60)
    
    print("❌ 잘못된 이해: 'Cumulative = 전체 대화'")
    print("✅ 올바른 이해:")
    print()
    
    print("📝 FORWARD-ONLY (미래 정보 차단):")
    print("  - 현재 시점 t에서 미래 정보 절대 사용 안 함")
    print("  - 사용 범위: [t-k+1, t-k+2, ..., t-1, t]")
    print("  - 실제 대화 상황과 동일")
    print()
    
    print("📚 CUMULATIVE (누적적 맥락):")
    print("  - Forward-only 범위 내에서 누적적으로 더 많은 context 사용")
    print("  - 고정 K 대신 상황에 따라 동적으로 조정")
    print("  - 예: 초기엔 3개, 후반엔 20개 (모두 forward-only)")
    print()
    
    print("🎯 Position Adaptive = Forward-only + Cumulative")
    print("  - Forward-only: 미래 정보 차단 ✅")
    print("  - Cumulative: 상황별 동적 context 크기 ✅")


def main():
    print("🎭 올바른 FORWARD-ONLY CUMULATIVE CONTEXT 설명")
    print("=" * 80)
    print("❗ 중요: Position Adaptive도 forward-only입니다!")
    print("차이점은 context 크기가 동적으로 바뀐다는 것입니다.")
    
    compare_strategies_correctly()
    show_cumulative_vs_forward_only()
    
    print("\n🎯 결론:")
    print("- Baseline: 항상 최근 K=10개 (forward-only)")
    print("- Position Adaptive: 위치별로 다른 크기 (forward-only)")
    print("- 둘 다 미래 정보는 절대 사용하지 않음!")
    print("- 차이는 context 크기가 고정 vs 동적인 것!")


if __name__ == "__main__":
    main()