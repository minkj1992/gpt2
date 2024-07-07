class Transformer(nn.Module):

    def __init__(
        self,
        encoder,
        decoder,
        source_pad_idx,
        target_pad_idx,
        device,
    ):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.source_pad_index = source_pad_idx
        self.target_pad_index = target_pad_idx
        self.device = device

    def get_source_mask(self, source):
        # source  = (bsize, source_len)
        # 0: pad, 1: value
        source_mask = source != self.source_pad_index

        # unsqueeze(i) operation adds an i_th dimension to the item
        source_mask = source_mask.unsqueeze(1)  # (bsize,1,source_len)
        source_mask = source_mask.unsqueeze(2)  # (bsize,1,1,source_len)

        # source_mask will be applied after calculating-
        # the attention which has 4 dimensions (bsize,n_attn_heads,qlen,klen)
        return source_mask

    def get_target_mask(self, target):
        # target = (bsize,target_len)
        target_length = target.shape[1]

        # Target mask = pad mask + look-ahead mask
        # (bsize,1,1,target_len)
        target_pad_mask = (target != self.target_pad_index).unsqueeze(1).unsqueeze(2)

        # tril = 행렬의 하부 삼각 부분
        target_subsequent_mask = torch.tril(
            torch.ones((target_length, target_length), device=self.device)
        ).bool()  # (target_len, target_len)

        return target_subsequent_mask & target_pad_mask  # (bsize,1,tlen,tlen)

    def forward(self, source, target):
        # source = (bsize,slen)
        # target = (bsize,tlen)

        source_mask = self.get_source_mask(source)  # (bsize,1,1,slen)
        target_mask = self.get_target_mask(target)  # (bsize,1,tlen,tlen)

        encoder_output = self.encoder(source, source_mask)  # (bsize,slen,hdim)
        decoder_output, decoder_attention = self.decoder(
            target,
            encoder_output,
            source_mask,
            target_mask,
        )
        # decoder_output = (bsiz,tlen,output_dim)
        # decoder_attention = (bsize,num_attn_heads,tlen,slen)
        return decoder_output, decoder_attention
