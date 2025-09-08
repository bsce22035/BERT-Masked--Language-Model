import sys
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, BertForMaskedLM

# Model & tokenizer
MODEL = "bert-base-uncased"
K = 15  # number of raw candidates to consider
FONT_PATH = "C:/Windows/Fonts/arial.ttf"  # adjust to a font on your system

try:
    FONT = ImageFont.truetype(FONT_PATH, 28)
except OSError:
    print("Warning: Could not load font. Using default.")
    FONT = ImageFont.load_default()

GRID_SIZE = 40
PIXELS_PER_WORD = 200

# Stop words for filtering
STOP_WORDS = {"it", "this", "that", "he", "she", "they", "him", "her",
              "and", "you", ".", ";", "!", "a", "the"}

def main():
    text = input("Enter a sentence with '[MASK]': ")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="pt")

    # Find [MASK] index
    mask_token_index = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if mask_token_index.numel() == 0:
        print(f"⚠️ Error: Input must contain {tokenizer.mask_token}")
        sys.exit(1)

    model = BertForMaskedLM.from_pretrained(MODEL, output_attentions=True)
    with torch.no_grad():
        result = model(**inputs)

    # Get predictions
    mask_token_logits = result.logits[0, mask_token_index.item(), :]
    top_tokens = torch.topk(mask_token_logits, K).indices.tolist()

    predictions = [
        tokenizer.decode([token]).strip()
        for token in top_tokens
        if tokenizer.decode([token]).strip().isalpha()
        and tokenizer.decode([token]).strip().lower() not in STOP_WORDS
    ][:5]  # take top 5 meaningful

    if predictions:
        print("Predictions:", predictions)
        for p in predictions:
            print(text.replace(tokenizer.mask_token, p))

    # Visualize attentions
    visualize_attentions(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]), result.attentions)


def get_color_for_attention_score(attention_score):
    """Convert attention score to grayscale color."""
    assert 0.0 <= attention_score <= 1.0
    gray_value = int(attention_score * 255)
    return (gray_value, gray_value, gray_value)

def visualize_attentions(tokens, attentions):
    for layer_index, layer_attentions in enumerate(attentions):
        for head_index, head_attentions in enumerate(layer_attentions[0]):
            generate_diagram(layer_index + 1, head_index + 1, tokens, head_attentions)

def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """Generates attention heatmap."""
    attention_weights = attention_weights.detach().cpu().numpy()
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)
    
    for i, token in enumerate(tokens):
        draw.text((PIXELS_PER_WORD - 10, PIXELS_PER_WORD + i * GRID_SIZE), token, fill="white", font=FONT)
        draw.text((image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE), token, fill="white", font=FONT)
    
    for i in range(len(tokens)):
        y = PIXELS_PER_WORD + i * GRID_SIZE
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            color = get_color_for_attention_score(float(attention_weights[i][j]))
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)
    
    img.save(f"Attention_Layer{layer_number}_Head{head_number}.png")

if __name__ == "__main__":
    main()
