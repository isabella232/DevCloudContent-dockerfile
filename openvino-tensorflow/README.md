# Openvino-Tensorflow
## Steps to run on Local

    docker build -t openvino-tensorflow .
    docker run -it --rm openvino-tensorflow
## Steps to run on Devcloud
1. Create Resource from Dockerfile
2. In GIT Repo URL field enter https://github.com/DevcloudContent/dockerfile-import.git
3. Enter a resource name
4. Click on Advanced Configuration
5. Leave Git Reference as blank
6. In Context Dir field enter openvino-tensorflow/
7. In Dockerfile Path field enter Dockerfile
8. Build the docker file. It might take sometime to build
9. Once Build/Image is ready assign a project and mention mount path as /app/result/ and launch it
10. Results will be availble in logs as well as filesystem
    
