import json
import itertools

def generate_configurations():
    apply_word_pe_options = [False, True]
    pooling_methods = ['lstm', 'self_attention', 'weighted_mean', 'simple_mean']
    apply_sentence_pe_options = [False, True]
    combination_methods = ['cross_attention_concatenate', 'cross_attention_sum', 'cross_attention_average', 'cross_attention_weighted', 'sum', 'concatenate']
    bayesian_methods = ['lstm', 'transformer', 'context_lstm']

    configurations = []
    config_id = 0

    print(f"Total combinations before processing: {len(list(itertools.product(apply_word_pe_options, pooling_methods, apply_sentence_pe_options, combination_methods, bayesian_methods)))}")

    for apply_word_pe, pooling_method, apply_sentence_pe, combination_method, bayesian_method in itertools.product(
        apply_word_pe_options, pooling_methods, apply_sentence_pe_options, combination_methods, bayesian_methods
    ):
        print(f"Processing: apply_word_pe={apply_word_pe}, pooling_method={pooling_method}, apply_sentence_pe={apply_sentence_pe}, combination_method={combination_method}, bayesian_method={bayesian_method}")
        
        if combination_method == 'cross_attention_weighted':
            for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
                config = {
                    "id": config_id,
                    "apply_word_pe": apply_word_pe,
                    "pooling_method": pooling_method,
                    "apply_sentence_pe": apply_sentence_pe,
                    "combination_method": combination_method,
                    "bayesian_method": bayesian_method,
                    "alpha": alpha
                }
                configurations.append(config)
                config_id += 1
                print(f"Added configuration with alpha={alpha}")
        else:
            config = {
                "id": config_id,
                "apply_word_pe": apply_word_pe,
                "pooling_method": pooling_method,
                "apply_sentence_pe": apply_sentence_pe,
                "combination_method": combination_method,
                "bayesian_method": bayesian_method
            }
            configurations.append(config)
            config_id += 1
            print("Added configuration")

    return configurations

def save_configurations(configurations, file_path='configurations.json'):
    with open(file_path, 'w') as f:
        json.dump({"configurations": configurations}, f, indent=2)

if __name__ == "__main__":
    configs = generate_configurations()
    save_configurations(configs)
    print(f"Generated {len(configs)} configurations and saved to configurations.json")
    
    print("\nExample configurations:")
    for i in range(min(5, len(configs))):
        print(f"\nConfiguration {configs[i]['id']}:")
        for key, value in configs[i].items():
            if key != 'id':
                print(f"  {key}: {value}")
