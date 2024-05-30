# ITV Speech Classification Number

## Install
- B1: Sử dụng minconda khởi tạo 1 python env với python version = 3.10.
1. `conda create -n cls_num_env python=3.10 -y`
- B2: Chạy lệnh sau để kích hoạt môi trường cls_num_env.
2. `conda activate cls_num_env`
- B3: cài đặt các thư viện cần thiết cho dự án.
3. `pip install -r requirements.txt`

## Dữ liệu

## Train
- B1: Chỉnh sửa các tham số training và đường dẫn data trong file *./configs/default.yaml*
- B2: Chỉnh sửa `pre_trained_model` trong *./configs/default.yaml*. Giá trị này có thể điền theo 2 cách:
    - Tên của pretrain model theo format của huggingface. VD: *vinai/PhoWhisper-tiny*
    - Đường dẫn đến pretrain model folder trên local. VD: *./exp/models/best"*
- B3: Gõ lệnh sau để bắt đầu quá trình train
```python main.py train --config default.yaml --outfile log.log```
Mô hình tốt nhất được lưu tại *./exp/model/best*

Trong đó:
--config: đường dẫn đến file config chứa các tham số của mô hình. Lưu ý, chỉ tên file, không phải đường dẫn. Mặc định: “default.yaml”
--outfile: đường dẫn đến file log để lưu lại output của quá trình training. Lưu ý, chỉ tên file, không phải đường dẫn. Mặc định: “log.log””

## Server
Chỉnh sửa service.checkpoint_dir trong file *./config/default.yaml* sau đó chạy câu lệnh sau để chạy server
```python main.py serve --config default.yaml```

### API
** http://127.0.0.1:8000/docs **
** http://127.0.0.1:8000/docs#/default/infer_cls_number_infer_post**

### API Documents
- **Enpoint:** /cls_number/infer

- **Header:**
{"accept: "application/json", "Content-Type": "application/json"}

- **Request body:**
```
{
  "input_audio": "string",
  "sample_rate": 0
}
```
input_audio: Union(str, torch.Tensor) -> có thể là đường dẫn của audio dưới dạng string hoặc waveform (sau khi torch.load) dưới dạng torch.Tensor
sample_rate: int -> mặc định là None, chỉ được sử dụng khi input_audio là waveform

- **Response body:**
```
{
  "labels": "string"
}
```
labels: str -> kết quả gắn nhãn cho input_audio sau khi inference qua mô hình

### Call API
Thay đổi audio_input ở dòng x trong file *inference.py* để request API bằng code python






