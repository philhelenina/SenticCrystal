"""
구체적인 예시로 Cumulative Context 설명
=====================================

실제 IEMOCAP 대화를 사용해서 다양한 context 전략이 어떻게 작동하는지 보여줍니다.
"""

# 실제 대화 예시: Ses01F_impro01 (ID 관련 대화)
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
    {"pos": 23, "speaker": "F", "text": "That's out of control.", "label": "ang"},  # 🎯 이걸 예측하려고 함!
]

def show_baseline_context(target_position=23):
    """Baseline: 고정 K=10 context"""
    print(f"🔍 BASELINE (고정 K=10): 위치 {target_position}에서 'ang' 예측")
    print("=" * 60)
    
    # 고정 K=10이므로 최근 10개 발화 사용
    start_idx = max(0, target_position - 10 + 1)
    context = dialogue_example[start_idx:target_position + 1]
    
    print(f"Context 크기: {len(context)} (고정)")
    print("사용되는 발화들:")
    
    for i, utt in enumerate(context):
        marker = "🎯" if utt["pos"] == target_position else "📝"
        label_info = f"[{utt['label']}]" if utt["label"] != "-1" else "[unlabeled]"
        print(f"  {marker} pos={utt['pos']:2d} {utt['speaker']}: {utt['text'][:50]}... {label_info}")
    
    labeled_count = sum(1 for utt in context if utt["label"] != "-1")
    print(f"\n레이블 있는 발화: {labeled_count}/{len(context)}")
    print(f"Context에서 감정 정보: {[utt['label'] for utt in context if utt['label'] != '-1']}")


def show_position_adaptive_context(target_position=23):
    """Position Adaptive: 대화 위치에 따라 context 크기 조정"""
    print(f"\n🧠 POSITION ADAPTIVE: 위치 {target_position}에서 'ang' 예측")
    print("=" * 60)
    
    dialogue_length = len(dialogue_example)
    rel_position = target_position / (dialogue_length - 1)
    
    # Position adaptive 로직
    if rel_position <= 0.1:      # 초기 10%
        max_context = 3
    elif rel_position <= 0.3:    # 초기 30%
        max_context = 8
    elif rel_position <= 0.7:    # 중간 70% 
        max_context = 15
    else:                        # 후반 30%
        max_context = 20
    
    actual_context = min(max_context, target_position + 1)
    
    print(f"대화 길이: {dialogue_length}, 상대적 위치: {rel_position:.2f}")
    print(f"위치 기반 최대 context: {max_context}")
    print(f"실제 context 크기: {actual_context}")
    
    start_idx = max(0, target_position - actual_context + 1)
    context = dialogue_example[start_idx:target_position + 1]
    
    print("\n사용되는 발화들:")
    for i, utt in enumerate(context):
        marker = "🎯" if utt["pos"] == target_position else "📝"
        label_info = f"[{utt['label']}]" if utt["label"] != "-1" else "[unlabeled]"
        print(f"  {marker} pos={utt['pos']:2d} {utt['speaker']}: {utt['text'][:50]}... {label_info}")
    
    labeled_count = sum(1 for utt in context if utt["label"] != "-1")
    print(f"\n레이블 있는 발화: {labeled_count}/{len(context)}")
    print(f"Context에서 감정 정보: {[utt['label'] for utt in context if utt['label'] != '-1']}")


def show_different_positions():
    """다양한 위치에서 어떻게 context가 달라지는지 보여줌"""
    print(f"\n📊 다양한 위치에서의 CONTEXT 비교")
    print("=" * 80)
    
    test_positions = [2, 9, 16, 23]  # 초기, 중기, 후기, 마지막
    
    for pos in test_positions:
        dialogue_length = len(dialogue_example)
        rel_position = pos / (dialogue_length - 1)
        
        # Baseline (고정 K=10)
        baseline_context = min(10, pos + 1)
        
        # Position adaptive
        if rel_position <= 0.1:
            adaptive_context = min(3, pos + 1)
        elif rel_position <= 0.3:
            adaptive_context = min(8, pos + 1)
        elif rel_position <= 0.7:
            adaptive_context = min(15, pos + 1)
        else:
            adaptive_context = min(20, pos + 1)
        
        current_label = dialogue_example[pos]["label"]
        current_text = dialogue_example[pos]["text"][:30]
        
        print(f"\n위치 {pos:2d} ({rel_position:.2f}): '{current_text}...' [{current_label}]")
        print(f"  Baseline K=10    : context={baseline_context}")
        print(f"  Position Adaptive: context={adaptive_context}")


def show_training_vs_context():
    """Training과 Context 사용의 차이점 설명"""
    print(f"\n🔄 TRAINING vs CONTEXT 사용법")
    print("=" * 60)
    
    print("📚 CONTEXT 생성 (ALL utterances 사용):")
    print("  - 레이블 있는 발화: context + training에 사용")
    print("  - 레이블 없는 발화 (-1): context에만 사용")
    print("  - 목적: 대화의 전체 흐름과 맥락 파악")
    
    print("\n🎯 TRAINING (labeled utterances만 사용):")
    print("  - 레이블 있는 발화만: 실제 모델 학습")
    print("  - 레이블 없는 발화 (-1): 학습에서 제외")
    print("  - 목적: 감정 분류 성능 향상")
    
    # 예시 계산
    total_utterances = len(dialogue_example)
    labeled_utterances = sum(1 for utt in dialogue_example if utt["label"] != "-1")
    
    print(f"\n📊 이 대화 예시에서:")
    print(f"  전체 발화: {total_utterances}개")
    print(f"  레이블 있는 발화: {labeled_utterances}개 ({labeled_utterances/total_utterances*100:.1f}%)")
    print(f"  레이블 없는 발화: {total_utterances-labeled_utterances}개 ({(total_utterances-labeled_utterances)/total_utterances*100:.1f}%)")
    
    print(f"\n💡 Context에는 모든 {total_utterances}개 발화 사용")
    print(f"💡 Training에는 {labeled_utterances}개 발화만 사용")


def main():
    print("🎭 CUMULATIVE CONTEXT 전략 구체적 예시")
    print("=" * 80)
    print("실제 IEMOCAP 대화: 'ID 문제로 점점 화나는 상황'")
    print("🎯 목표: 마지막 발화 'That's out of control.'의 감정 'ang' 예측")
    
    # 1. Baseline 방식
    show_baseline_context()
    
    # 2. Position Adaptive 방식  
    show_position_adaptive_context()
    
    # 3. 다양한 위치 비교
    show_different_positions()
    
    # 4. Training vs Context 설명
    show_training_vs_context()
    
    print(f"\n🎯 결론:")
    print(f"- Baseline: 항상 최근 10개만 봄 (단순)")
    print(f"- Position Adaptive: 대화 후반부에서 더 많은 context 활용 (똑똑함)")
    print(f"- 레이블 없어도 대화 흐름 파악에는 중요!")
    print(f"- 화가 나는 과정을 더 잘 이해할 수 있음")


if __name__ == "__main__":
    main()