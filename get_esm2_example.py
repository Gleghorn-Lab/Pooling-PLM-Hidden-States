if __name__ == '__main__':
    import torch
    from torchinfo import summary
    from ESMplusplus.modeling_fastesm import FastEsmForMaskedLM

    # Load model and tokenizer
    model_path = 'facebook/esm2_t33_650M_UR50D'
    model = FastEsmForMaskedLM.from_pretrained(model_path).eval()
    tokenizer = model.tokenizer

    # Test forward
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_ids = tokenizer.encode('MILK', return_tensors='pt')
    summary(model, input_data=input_ids, device=device)

