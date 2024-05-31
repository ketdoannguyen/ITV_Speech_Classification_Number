# ITV Speech Classification Number

## Install

- B1: Sử dụng minconda khởi tạo 1 python env với python version = 3.10.
1. `conda create -n cls_num_env python=3.10 -y`
- B2: Chạy lệnh sau để kích hoạt môi trường cls_num_env.
2. `conda activate cls_num_env`
- B3: cài đặt các thư viện cần thiết cho dự án.
3. `pip install -r requirements.txt`

## Dữ liệu

### Split train test

Chạy lệnh sau để split train set data
`python main.py split_train_test --config default.yaml test_split 0.2`


### Augment data

- B1: Chỉnh sửa các tham số ở phần augment trong đường dẫn file _./configs/default.yaml_. Ví dụ: **change_pitch: [-0.5, 1]** thì [-0.5, 1] là 2 con số tương ứng lần lược là pitch giảm và pitch tăng. Bạn có thể điều chỉnh các con số đó phù hợp data của mình
- B2: Chỉnh sửa các tham số ở phần data trong đường dẫn file _./configs/default.yaml_. Gồm các đường dẫn file csv, folder lưu data augment.
- B3: Gõ lệnh sau để bắt đầu quá trình tăng cường dữ liệu, mặc định dữ liệu tăng cường được lưu tại *./dataset/train_aug*
  `python main.py augment --config default.yaml`

## Train

- B1: Chỉnh sửa các tham số training và đường dẫn data trong file _./configs/default.yaml_
- B2: Chỉnh sửa `pre_trained_model` trong _./configs/default.yaml_. Giá trị này có thể điền theo 2 cách:
  - Tên của pretrain model theo format của huggingface. VD: _vinai/PhoWhisper-tiny_
  - Đường dẫn đến pretrain model folder trên local. VD: _./exp/models/best"_
- B3: Gõ lệnh sau để bắt đầu quá trình train
  `python main.py train --config default.yaml --outfile log.log --is_aug True`
  Mô hình tốt nhất được lưu tại _./exp/model/best_

Trong đó:
--config: đường dẫn đến file config chứa các tham số của mô hình. Lưu ý, chỉ tên file, không phải đường dẫn. Mặc định: “default.yaml”
--outfile: đường dẫn đến file log để lưu lại output của quá trình training. Lưu ý, chỉ tên file, không phải đường dẫn. Mặc định: “log.log””
--is_aug: có tăng cường dữ liệu trước khi training hay không. Mặc định True

## Server

Chỉnh sửa service.checkpoint*dir trong file *./config/default.yaml\_ sau đó chạy câu lệnh sau để chạy server
`python main.py serve --config default.yaml`

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

Thay đổi audio*input ở dòng x trong file \_inference.py* để request API bằng code python
