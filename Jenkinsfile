pipeline {
  agent none
  options {
    disableConcurrentBuilds()
    buildDiscarder(logRotator(numToKeepStr: '8', daysToKeepStr: '20'))
    timeout(time: 1, unit: 'HOURS')
  }
  environment {
    IMAGE = "$REGISTRY_PREFIX/${JOB_NAME.toLowerCase()}:$BUILD_NUMBER"
    PARALLEL = "4"
  }
  stages {
    stage('main') {
      parallel {
        stage('cpu') {
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
            stage('build') {
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
                sh '''
                  cd $BUILD && cmake $SRC \
                    -DCMAKE_BUILD_TYPE=Release \
                    -DCMAKE_INSTALL_PREFIX="." \
                    -DCOMPILE_NDA_TESTS=OFF \
                    -DENABLE_FFTW=ON \
                    -DENABLE_CPPTRACE=OFF \
                    -DENABLE_SPDLOG=ON \
                    -DCTEST_NPROC=$PARALLEL
                '''
                sh 'make -C $BUILD -j $PARALLEL'
                warnError("Tests failed") {
                  sh 'cd $BUILD && ctest --output-on-failure'
                }
              }
            }
            stage('asan_and_ubsan') {
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
                sh '''
                  cd $BUILD && cmake $SRC \
                    -DCMAKE_BUILD_TYPE=Debug \
                    -DCMAKE_INSTALL_PREFIX="." \
                    -DCOMPILE_NDA_TESTS=OFF \
                    -DENABLE_FFTW=ON \
                    -DENABLE_CPPTRACE=OFF \
                    -DENABLE_SPDLOG=ON \
                    -DENABLE_ASAN=ON \
                    -DENABLE_UBSAN=ON \
                    -DCTEST_NPROC=$PARALLEL
                '''
                sh 'make -C $BUILD -j $PARALLEL'
                warnError("Tests on Debug with ASAN and UBSAN failed") {
                  sh 'cd $BUILD && ctest --output-on-failure'
                }
              }
            }
          }
        }
        stage('cuda') {
          stages {
            stage('image') {
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
                      imagePullSecrets:
                        - name: registry-auth
                      containers:
                        - name: main
                          image: $IMAGE-cuda
                          command: [sleep]
                          args: [99999]
                          securityContext:
                            runAsUser: 1000
                            runAsGroup: 1000
                          resources:
                            limits:
                              cpu: $PARALLEL
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
                sh '''
                  cd $BUILD && cmake $SRC \
                    -DCMAKE_BUILD_TYPE=Release \
                    -DCMAKE_INSTALL_PREFIX="." \
                    -DCOMPILE_NDA_TESTS=OFF \
                    -DENABLE_FFTW=ON \
                    -DENABLE_CPPTRACE=OFF \
                    -DENABLE_SPDLOG=ON \
                    -DCTEST_NPROC=4 \
                    -DENABLE_CUDA=ON
                '''
                sh 'make -C $BUILD -j $PARALLEL'
                warnError("Tests failed") {
                  sh 'cd $BUILD && ctest --output-on-failure'
                }
              }
            }
          }
        }
      }
    }
  }
}
