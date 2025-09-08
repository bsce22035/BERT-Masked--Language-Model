# Masked Language Model + Attention Visualization

This project uses a **BERT masked language model** to:
- Predict missing words in a sentence containing `[MASK]`
- Visualize the model’s **attention weights** as heatmaps

Built with **PyTorch**, Hugging Face **Transformers**, and **Pillow**.

---

## Example
Enter a sentence with '[MASK]': The capital of France is [MASK].

## Output

Console:

Predictions: ['Paris', 'Lyon', 'Marseille', 'Berlin', 'London']
The capital of France is Paris.
The capital of France is Lyon.
The capital of France is Marseille.
The capital of France is Berlin.
The capital of France is London.

Attention heatmaps saved as PNG files (Attention_Layer1_Head1.png, etc.)

