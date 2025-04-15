from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
 
# Reference and generated texts
reference_texts = [
    ["The cat sat on the mat.", "A cat is sitting on the mat."],
]
generated_texts = [
    "The lion is on my lap.",
]
 
# Function to calculate BLEU
def calculate_bleu(generated, references):
    scores = []
    for gen, refs in zip(generated, references):
        refs_tokenized = [ref.split() for ref in refs]
        gen_tokenized = gen.split()
        score = sentence_bleu(refs_tokenized, gen_tokenized)
        scores.append(score)
    return scores
 
 
# Function to calculate ROUGE-1
def calculate_rouge1(generated, references):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = []
    for gen, refs in zip(generated, references):
        # Evaluate against all references and take the highest score
        rouge1_scores = [scorer.score(ref, gen)['rouge1'] for ref in refs]
        max_rouge1 = max(rouge1_scores, key=lambda x: x.fmeasure)
        scores.append(max_rouge1)
    return scores
 
# Calculate BLEU and ROUGE-1
bleu_scores = calculate_bleu(generated_texts, reference_texts)
rouge1_scores = calculate_rouge1(generated_texts, reference_texts)
 
# Display Results
print("BLEU Scores:", bleu_scores)
print("ROUGE-1 Scores (Precision, Recall, F1):", [(s.precision, s.recall, s.fmeasure) for s in rouge1_scores])