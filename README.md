# BASNet HTTP

This is an HTTP service wrapper for [BASNet: Boundary-Aware Salient Object Detection code](https://github.com/NathanUA/BASNet)

The deploy folder contains configuration files for deployment as serverless container with Knative.

It's highly recommended to run this image on a machine with a CUDA compatible Nvidia Card and minimum 6Gb of RAM.

# Usage:

```bash
docker run --rm -p 8080:80 docker.io/cyrildiagne/basnet-http
```

If you're having `empty response` issues, make sure your docker instance has 6gb of RAM as mentioned [here](https://github.com/cyrildiagne/ar-cutpaste/issues/26#issuecomment-625689672)

# Test:

```bash
curl -F "data=@test.jpg" http://localhost:8080 -o result.png
```

# Deploy on Google App Engine

Create an App Engine application. [Instructions](https://cloud.google.com/appengine/docs/flexible/custom-runtimes/managing-projects-apps-billing#create)

Deploy your code:
```bash
gcloud app deploy --project=<YOUR_GCP_PROJECT>
```

# Development

- Clone this repository: `git clone https://github.com/cyrildiagne/BASNet-http.git`
- Go into the cloned directory: `cd BASNet-http`
- Clone the [BASNet repository](https://github.com/NathanUA/BASNet)
- Download the pretrained model [basnet.pth](https://drive.google.com/open?id=1s52ek_4YTDRt_EOkx1FS53u-vJa0c4nu)
- Put the file inside the `BASNet/saved_models/basnet_bsi/` folder.

# Build from source:

### Option 1 - Locally with virtualenv

Requires Python v3.6+

```bash
virtualenv venv
venv/bin/activate
```

```bash
pip install torch==0.4.1
pip install -r requirements.txt
```

```
python main.py
```

### Option 2 - Using Docker

After you've retrieved the BASNet model.

Download Resnet checkpoint
```
curl https://download.pytorch.org/models/resnet34-333f7ec4.pth -o resnet34-333f7ec4.pth
```

```
docker build -t basnet .
docker run --rm -p 8080:80 basnet
```
