"""
Adaptive Cumulative Context Strategy Design
==========================================

Design intelligent context sizing based on dialogue position and patterns.
All utterances (including -1 labels) are used for context, but only labeled ones for training.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HOME_DIR = Path(__file__).parent.parent
RESULTS_DIR = HOME_DIR / 'results' / 'adaptive_context'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class AdaptiveContextStrategy:
    """Adaptive context sizing based on dialogue position and characteristics."""
    
    def __init__(self, dialogue_stats):
        self.dialogue_stats = dialogue_stats
        self.avg_dialogue_length = dialogue_stats['mean_length']
        self.median_length = dialogue_stats['median_length']
        
    def position_based_context(self, position, dialogue_length):
        """Context size based on position in dialogue."""
        
        # Relative position in dialogue (0.0 = start, 1.0 = end)
        rel_position = position / max(dialogue_length - 1, 1)
        
        if rel_position <= 0.1:  # Very early (first 10%)
            return min(3, position + 1)  # Small context
        elif rel_position <= 0.3:  # Early (10-30%)
            return min(8, position + 1)  # Growing context
        elif rel_position <= 0.7:  # Middle (30-70%)
            return min(15, position + 1)  # Full context
        else:  # Late (70-100%)
            return min(20, position + 1)  # Maximum context
    
    def emotion_density_context(self, dialogue_emotions, position):
        """Adjust context based on emotion density in recent history."""
        
        # Look at emotion density in last 10 utterances
        start_idx = max(0, position - 10)
        recent_emotions = dialogue_emotions[start_idx:position+1]
        
        # Count non-null emotions
        emotion_count = sum(1 for e in recent_emotions if e != '-1')
        emotion_density = emotion_count / len(recent_emotions) if recent_emotions else 0
        
        base_context = self.position_based_context(position, len(dialogue_emotions))
        
        if emotion_density > 0.7:  # High emotion density
            return min(25, base_context + 5)  # Extend context
        elif emotion_density < 0.3:  # Low emotion density  
            return max(3, base_context - 2)  # Reduce context
        else:
            return base_context
    
    def speaker_change_context(self, dialogue_speakers, position):
        """Adjust context based on speaker change patterns."""
        
        if position == 0:
            return 1
        
        # Count speaker changes in recent history
        start_idx = max(0, position - 5)
        recent_speakers = dialogue_speakers[start_idx:position+1]
        
        changes = 0
        for i in range(1, len(recent_speakers)):
            if recent_speakers[i] != recent_speakers[i-1]:
                changes += 1
        
        change_rate = changes / max(len(recent_speakers) - 1, 1)
        
        base_context = self.position_based_context(position, len(dialogue_speakers))
        
        if change_rate > 0.8:  # Rapid turn-taking
            return min(12, base_context + 2)  # Moderate increase
        elif change_rate < 0.4:  # Slow turn-taking (long turns)
            return min(20, base_context + 5)  # Bigger increase
        else:
            return base_context
    
    def combined_adaptive_context(self, dialogue_data, position):
        """Combine all adaptive strategies."""
        
        dialogue_length = len(dialogue_data)
        dialogue_emotions = [row['label'] for row in dialogue_data]
        dialogue_speakers = [row['speaker'] for row in dialogue_data]
        
        # Get context sizes from different strategies
        pos_context = self.position_based_context(position, dialogue_length)
        emotion_context = self.emotion_density_context(dialogue_emotions, position)
        speaker_context = self.speaker_change_context(dialogue_speakers, position)
        
        # Weighted combination
        combined = int(0.4 * pos_context + 0.3 * emotion_context + 0.3 * speaker_context)
        
        # Ensure reasonable bounds
        final_context = max(1, min(25, combined))
        
        return {
            'final_context': final_context,
            'position_based': pos_context,
            'emotion_based': emotion_context,
            'speaker_based': speaker_context,
            'dialogue_position': position / max(dialogue_length - 1, 1)
        }


def design_cumulative_strategies():
    """Design comprehensive cumulative context strategies."""
    
    logger.info("Designing cumulative context strategies...")
    
    strategies = {
        'fixed_short': {
            'type': 'fixed',
            'context_size': 5,
            'description': 'Fixed short context (5 utterances)',
            'use_case': 'Quick emotional reactions'
        },
        
        'fixed_medium': {
            'type': 'fixed', 
            'context_size': 10,
            'description': 'Fixed medium context (10 utterances)',
            'use_case': 'Balanced context/computation'
        },
        
        'fixed_long': {
            'type': 'fixed',
            'context_size': 20,
            'description': 'Fixed long context (20 utterances)', 
            'use_case': 'Rich contextual understanding'
        },
        
        'position_adaptive': {
            'type': 'adaptive',
            'function': 'position_based_context',
            'description': 'Context grows with dialogue position',
            'use_case': 'Natural dialogue progression'
        },
        
        'emotion_adaptive': {
            'type': 'adaptive',
            'function': 'emotion_density_context', 
            'description': 'Context adapts to emotion density',
            'use_case': 'Emotion-aware modeling'
        },
        
        'speaker_adaptive': {
            'type': 'adaptive',
            'function': 'speaker_change_context',
            'description': 'Context adapts to turn-taking patterns',
            'use_case': 'Turn-structure aware modeling'
        },
        
        'fully_adaptive': {
            'type': 'adaptive',
            'function': 'combined_adaptive_context',
            'description': 'Combines position, emotion, and speaker patterns',
            'use_case': 'Maximum contextual intelligence'
        }
    }
    
    return strategies


def simulate_adaptive_contexts(dialogue_data, strategies):
    """Simulate how different strategies would work on sample dialogues."""
    
    logger.info("Simulating adaptive context strategies...")
    
    # Load dialogue stats from previous analysis
    stats_file = HOME_DIR / 'results' / 'dialogue_analysis' / 'comprehensive_dialogue_analysis.json'
    with open(stats_file) as f:
        analysis = json.load(f)
    
    train_stats = analysis['train']['length_stats']
    
    # Initialize adaptive strategy
    adaptive = AdaptiveContextStrategy(train_stats)
    
    simulation_results = {}
    
    for strategy_name, strategy_config in strategies.items():
        logger.info(f"Simulating strategy: {strategy_name}")
        
        context_sizes = []
        positions = []
        
        # Simulate on sample dialogues
        for dialogue_id, dialogue_group in dialogue_data.groupby('dialogue_id'):
            dialogue_length = len(dialogue_group)
            
            # Convert to list of dicts for adaptive strategy
            dialogue_list = []
            for _, row in dialogue_group.iterrows():
                dialogue_list.append({
                    'label': row['label'],
                    'speaker': row['speaker'],
                    'position': row['utt_num_in_dialogue']
                })
            
            for position in range(dialogue_length):
                positions.append(position / max(dialogue_length - 1, 1))  # Relative position
                
                if strategy_config['type'] == 'fixed':
                    context_size = min(strategy_config['context_size'], position + 1)
                else:
                    # Adaptive strategy
                    if strategy_config['function'] == 'position_based_context':
                        context_size = adaptive.position_based_context(position, dialogue_length)
                    elif strategy_config['function'] == 'emotion_density_context':
                        context_size = adaptive.emotion_density_context(
                            [d['label'] for d in dialogue_list], position
                        )
                    elif strategy_config['function'] == 'speaker_change_context':
                        context_size = adaptive.speaker_change_context(
                            [d['speaker'] for d in dialogue_list], position
                        )
                    elif strategy_config['function'] == 'combined_adaptive_context':
                        result = adaptive.combined_adaptive_context(dialogue_list, position)
                        context_size = result['final_context']
                    else:
                        context_size = min(10, position + 1)  # Default
                
                context_sizes.append(context_size)
        
        simulation_results[strategy_name] = {
            'avg_context_size': np.mean(context_sizes),
            'std_context_size': np.std(context_sizes),
            'min_context_size': np.min(context_sizes),
            'max_context_size': np.max(context_sizes),
            'context_sizes': context_sizes[:1000],  # Sample for visualization
            'positions': positions[:1000]
        }
        
        logger.info(f"  {strategy_name}: avg={np.mean(context_sizes):.1f}, "
                   f"std={np.std(context_sizes):.1f}, "
                   f"range=[{np.min(context_sizes)}-{np.max(context_sizes)}]")
    
    return simulation_results


def create_strategy_visualizations(simulation_results, save_dir):
    """Create visualizations comparing different strategies."""
    
    logger.info("Creating strategy comparison visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink']
    
    for i, (strategy_name, results) in enumerate(simulation_results.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Plot context size vs position
        ax.scatter(results['positions'], results['context_sizes'], 
                  alpha=0.6, s=10, color=colors[i % len(colors)])
        ax.set_xlabel('Relative Position in Dialogue')
        ax.set_ylabel('Context Size')
        ax.set_title(f'{strategy_name}\nAvg: {results["avg_context_size"]:.1f}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 25)
    
    # Hide unused subplots
    for i in range(len(simulation_results), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'adaptive_strategies_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary comparison
    plt.figure(figsize=(12, 8))
    
    strategy_names = list(simulation_results.keys())
    avg_sizes = [results['avg_context_size'] for results in simulation_results.values()]
    std_sizes = [results['std_context_size'] for results in simulation_results.values()]
    
    bars = plt.bar(strategy_names, avg_sizes, yerr=std_sizes, capsize=5, alpha=0.7)
    plt.xlabel('Strategy')
    plt.ylabel('Average Context Size')
    plt.title('Context Size Comparison Across Strategies')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, avg in zip(bars, avg_sizes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{avg:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'strategy_comparison_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualizations saved to {save_dir}")


def main():
    """Main analysis function."""
    
    logger.info("="*80)
    logger.info("ADAPTIVE CUMULATIVE CONTEXT STRATEGY DESIGN")
    logger.info("="*80)
    
    # Load train data for simulation
    from pathlib import Path
    DATA_DIR = HOME_DIR / 'data' / 'iemocap_4way_data'
    
    train_file = DATA_DIR / 'train_4way_with_minus_one.csv'
    df = pd.read_csv(train_file)
    
    # Add dialogue info
    df['dialogue_id'] = df['file_id']
    df['speaker'] = df['id'].str.extract(r'Ses\d+[MF]_[^_]+_([MF])\d+')
    df['utt_num_in_dialogue'] = df['utterance_num']
    
    logger.info(f"Loaded {len(df)} utterances from {len(df['dialogue_id'].unique())} dialogues")
    
    # Design strategies
    strategies = design_cumulative_strategies()
    
    logger.info("\nDesigned strategies:")
    for name, config in strategies.items():
        logger.info(f"  {name}: {config['description']}")
    
    # Simulate strategies
    simulation_results = simulate_adaptive_contexts(df, strategies)
    
    # Create visualizations
    create_strategy_visualizations(simulation_results, RESULTS_DIR)
    
    # Save comprehensive results
    combined_results = {
        'strategies': strategies,
        'simulation_results': simulation_results,
        'recommendations': {
            'quick_prototyping': 'fixed_medium',
            'balanced_performance': 'position_adaptive', 
            'maximum_intelligence': 'fully_adaptive',
            'computational_efficiency': 'fixed_short'
        }
    }
    
    # Convert numpy types for JSON
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        else:
            return obj
    
    results_file = RESULTS_DIR / 'adaptive_context_strategies.json'
    with open(results_file, 'w') as f:
        json.dump(convert_numpy(combined_results), f, indent=2)
    
    logger.info(f"\nResults saved: {results_file}")
    
    # Print recommendations
    logger.info("\n" + "="*80)
    logger.info("ADAPTIVE CONTEXT STRATEGY RECOMMENDATIONS")
    logger.info("="*80)
    
    logger.info("Key Principles:")
    logger.info("1. ALL utterances (including -1 labels) used for CONTEXT")
    logger.info("2. Only labeled utterances used for TRAINING")
    logger.info("3. Context size adapts to dialogue characteristics")
    logger.info("4. Balance between context richness and computation")
    
    logger.info(f"\nStrategy Performance Summary:")
    for name, results in simulation_results.items():
        logger.info(f"  {name}: {results['avg_context_size']:.1f} ± {results['std_context_size']:.1f}")
    
    logger.info(f"\nRecommended Implementation Order:")
    logger.info(f"1. fixed_medium (K=10) - baseline comparison")
    logger.info(f"2. position_adaptive - smart scaling")
    logger.info(f"3. fully_adaptive - maximum intelligence")
    
    logger.info("\n✅ Adaptive context strategy design completed!")


if __name__ == "__main__":
    main()