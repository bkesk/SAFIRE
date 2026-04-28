pipeline {
  agent none
  options {
    disableConcurrentBuilds()
    buildDiscarder(logRotator(numToKeepStr: '8', daysToKeepStr: '20'))
    timeout(time: 1, unit: 'HOURS')
  }
  environment {
    IMAGE = "$REGISTRY_PREFIX/${JOB_NAME.toLowerCase()}:$BUILD_NUMBER"
  }
  stages {
    stage('image') {
      agent {
        kubernetes {
          inheritFrom 'podman'
          defaultContainer 'main'
        }
      }
      steps {
        sh 'podman build -t $IMAGE docker -f docker/Dockerfile_jenkins'
        sh 'podman push $IMAGE'
      }
    }
    stage('main') {
      agent {
        kubernetes {
          inheritFrom 'default'
          containerTemplate {
            name 'main'
            image IMAGE
          }
          defaultContainer 'main'
        }
      }
      environment {
        SRC = pwd()
        BUILD = pwd(tmp:true)
      }
      steps {
        sh 'cd $BUILD && cmake $SRC -DCMAKE_BUILD_TYPE=Release'
        sh 'make -C $BUILD -j $PARALLEL'
        warnError("Tests failed") {
          sh 'cd $BUILD && ctest --output-on-failure'
        }
      }
    }
    stage('cuda image') {
      agent {
        kubernetes {
          inheritFrom 'podman'
          defaultContainer 'main'
        }
      }
      steps {
        sh 'podman build -t $IMAGE-cuda docker -f docker/Dockerfile_jenkins_cuda'
        sh 'podman push $IMAGE-cuda'
      }
    }
    stage('cuda') {
      agent {
        kubernetes {
          yaml """
            spec:
              runtimeClassName: nvidia
              containers:
                - name: main
                  image: $IMAGE-cuda
                  command: [sleep]
                  args: [99999]
                  resources:
                    limits:
                      cpu: 4
                      memory: 16Gi
                      nvidia.com/gpu: 1
          """
          defaultContainer 'main'
        }
      }
      environment {
        SRC = pwd()
        BUILD = pwd(tmp:true)
      }
      steps {
        sh 'cd $BUILD && cmake $SRC -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON'
        sh 'make -C $BUILD -j $PARALLEL'
        warnError("Tests failed") {
          sh 'cd $BUILD && ctest --output-on-failure'
        }
      }
    }
  }
}

