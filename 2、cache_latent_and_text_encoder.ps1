# Cache script by @bdsqlsz

# Cache lantent
$dataset_config = "./toml/qinglong-datasets.toml"            # path to dataset config .toml file | 数据集配置文件路径
$vae = "./ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt" # VAE directory | VAE路径
$vae_dtype = ""                                              # fp16 | fp32 |bf16 default: fp16
$vae_chunk_size = 32                                         # chunk size for CausalConv3d in VAE
$vae_tiling = $True                                          # enable spatial tiling for VAE, default is False. If vae_spatial_tile_sample_min_size is set, this is automatically enabled
$vae_spatial_tile_sample_min_size = 256                      # spatial tile sample min size for VAE, default 256
$device = ""                                                 # cuda | cpu
$batch_size = ""                                             # batch size, override dataset config if dataset batch size > this
$num_workers = 0                                             # number of workers for dataset. default is cpu count-1
$skip_existing = $True                                       # skip existing cache files
$debug_mode = ""                                             # image | console
$console_width = $Host.UI.RawUI.WindowSize.Width             # console width
$console_back = "black"                                      # console background color
$console_num_images = 16                                     # number of images to show in console

# Cache text encoder
$text_encoder1 = "./ckpts/text_encoder/llava_llama3_fp16.safetensors"     # Text Encoder 1 directory | 文本编码器路径
$text_encoder2 = "./ckpts/text_encoder_2/clip_l.safetensors"              # Text Encoder 2 directory | 文本编码器路径
$text_encoder_batch_size = "16"                                           # batch size
$text_encoder_device = ""                                                 # cuda | cpu
$text_encoder_dtype = "bf16"                                              # fp16 | fp32 |bf16 default: fp16
$fp8_llm = $False                                                         # enable fp8 for text encoder
$text_encoder_num_workers = 0                                             # number of workers for dataset. default is cpu count-1
$text_encoder_skip_existing = $False                                       # skip existing cache files

# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================
# Activate python venv
Set-Location $PSScriptRoot
if ($env:OS -ilike "*windows*") {
  if (Test-Path "./venv/Scripts/activate") {
    Write-Output "Windows venv"
    ./venv/Scripts/activate
  }
  elseif (Test-Path "./.venv/Scripts/activate") {
    Write-Output "Windows .venv"
    ./.venv/Scripts/activate
  }
}
elseif (Test-Path "./venv/bin/activate") {
  Write-Output "Linux venv"
  ./venv/bin/Activate.ps1
}
elseif (Test-Path "./.venv/bin/activate") {
  Write-Output "Linux .venv"
  ./.venv/bin/activate.ps1
}

$Env:HF_HOME = "huggingface"
#$Env:HF_ENDPOINT = "https://hf-mirror.com"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
$ext_args = [System.Collections.ArrayList]::new()
$ext2_args = [System.Collections.ArrayList]::new()

if ($vae_chunk_size) {
  [void]$ext_args.Add("--vae_chunk_size=$vae_chunk_size")
}

if ($vae_tiling) {
  [void]$ext_args.Add("--vae_tiling")
}

if ($vae_dtype) {
  [void]$ext_args.Add("--vae_dtype=$vae_dtype")
}

if ($vae_spatial_tile_sample_min_size) {
  [void]$ext_args.Add("--vae_spatial_tile_sample_min_size=$vae_spatial_tile_sample_min_size")
}

if ($device) {
  [void]$ext_args.Add("--device=$device")
}

if ($batch_size) {
  [void]$ext_args.Add("--batch_size=$batch_size")
}

if ($num_workers -ne 0) {
  [void]$ext_args.Add("--num_workers=$num_workers")
}

if ($skip_existing) {
  [void]$ext_args.Add("--skip_existing")
}

if ($debug_mode) {
  [void]$ext_args.Add("--debug_mode=$debug_mode")
  if ($debug_mode -eq "console") {
    if ($console_width) {
      [void]$ext_args.Add("--console_width=$console_width")
    }
    
    if ($console_back) {
      [void]$ext_args.Add("--console_back=$console_back")
    }
    
    if ($console_num_images) {
      [void]$ext_args.Add("--console_num_images=$console_num_images")
    }
  }
}

if ($text_encoder_batch_size) {
  [void]$ext2_args.Add("--batch_size=$text_encoder_batch_size")
}

if ($text_encoder_device) {
  [void]$ext2_args.Add("--device=$text_encoder_device")
}

if ($text_encoder_dtype) {
  [void]$ext2_args.Add("--text_encoder_dtype=$text_encoder_dtype")
}

if ($fp8_llm) {
  [void]$ext2_args.Add("--fp8_llm")
}

if ($text_encoder_num_workers -ne 0) {
  [void]$ext2_args.Add("--num_workers=$text_encoder_num_workers")
}

if ($text_encoder_skip_existing) {
  [void]$ext2_args.Add("--skip_existing")
}

# run Cache
python "./musubi-tuner/cache_latents.py" `
  --dataset_config=$dataset_config `
  --vae=$vae $ext_args

python "./musubi-tuner/cache_text_encoder_outputs.py" `
  --dataset_config=$dataset_config `
  --text_encoder1=$text_encoder1 `
  --text_encoder2=$text_encoder2 $ext2_args

Write-Output "Cache finished"
Read-Host | Out-Null ;
