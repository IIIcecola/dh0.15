import torch
from ModelDecoder import TransformerStackedDecoder
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from PreProcess import ctrl_expressions as ctrl_expressions_list
from AudioDataset import WavSample


if __name__ == '__main__':
    print("___________")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("./wav2vec2-base-960h")

    wav_path = './Dataset/zhuboshuolianbo/wav/BV1A3H6z9Exx.wav'
    json_path = './Dataset/zhuboshuolianbo/json/CD_BV1A3H6z9Exx_1.json'


    testWavSample = WavSample(json_path=json_path,wav_path=wav_path,processor=processor,model=model)

    decoderModel = TransformerStackedDecoder(
        input_dim=768,
        output_dim=136
    ).to(device)

    state_dict = torch.load("./Weights/transformer_decoder.pth", map_location=device)
    decoderModel.load_state_dict(state_dict)
    decoderModel.to(device)
    decoderModel.eval()

    exp_list = ["CTRL_expressions_browDownL", "CTRL_expressions_browDownR"]
    mouth_ctrls = [name for name in ctrl_expressions_list if "mouth" in name]
    print(mouth_ctrls)
    print(len(mouth_ctrls))
    testWavSample.plot_compare(5,channel=mouth_ctrls,decoder=decoderModel)
