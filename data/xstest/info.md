task_categories:
  - text2text-generation
pretty_name: exaggerated safety
dataset_info:
  features:
    - name: prompt
      dtype: string
    - name: focus
      dtype: string
    - name: type
      dtype: string
    - name: note
      dtype: string
    - name: label
      dtype: string
  splits:
    - name: test
      num_bytes: 43841
      num_examples: 450
  download_size: 16964
  dataset_size: 43841
250 safe prompts across ten prompt types that well-calibrated models should not refuse to comply with.
200 unsafe prompts as contrasts that, for most LLM applications, should be refused.