# Zero-Shot Demo

This repo shows how to bucket any sentence into your custom labels using zero-shot AI.

## Run it locally

\`\`\`bash
python3 -m venv ai_env && source ai_env/bin/activate
pip install "numpy<2" torch transformers sentencepiece
python zero_shot.py \\
  --text "Your sentence here" \\
  --labels "Label1,Label2,Label3"
\`\`\`
