from transformers import T5Tokenizer, T5ForConditionalGeneration

class MultiDocSummarizer:
    def __init__(self, model_name='t5-base', max_length=512):
        """
        Initialize the T5 model and tokenizer.
        """
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.max_length = max_length

    def summarize(self, input_text, min_length=100, max_length=None):
        """
        Summarize the input text using the T5 model.
        """
        if max_length is None:
            max_length = self.max_length

        # Prepend "summarize: " to the input text (T5 expects this format)
        input_text = f"summarize: {input_text.strip()}"

        # Tokenize and summarize
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt', truncation=True)
        summary_ids = self.model.generate(
            input_ids=input_ids,
            num_beams=4,
            no_repeat_ngram_size=2,
            min_length=min_length,
            max_length=max_length,
            early_stopping=True
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)