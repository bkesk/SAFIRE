pipeline {
  agent none
  options {
    disableConcurrentBuilds()
    buildDiscarder(logRotator(numToKeepStr: '8', daysToKeepStr: '20'))
    timeout(time: 1, unit: 'HOURS')
  }
  stages {
    stage('main') {
      agent {
         dockerfile {
            dir 'docker'
            filename 'Dockerfile_jenkins'
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
    stage('cuda') {
      agent {
         dockerfile {
            dir 'docker'
            filename 'Dockerfile_jenkins_cuda'
            args '--gpus 1'
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

