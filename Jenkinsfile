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
        sh 'cd $BUILD && cmake $SRC -DCMAKE_BUILD_TYPE=Release'
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
        sh 'cd $BUILD && cmake $SRC -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON'
        sh 'make -C $BUILD -j $PARALLEL'
        warnError("Tests failed") {
          sh 'cd $BUILD && ctest --output-on-failure'
        }
      }
    }
  }
}

