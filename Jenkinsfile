properties([
  disableConcurrentBuilds(),
  buildDiscarder(logRotator(numToKeepStr: '8', daysToKeepStr: '20'))
])

timeout(time: 1, unit: 'HOURS') {
  parallel cpu: {
    buildPod(context: 'docker', dockerfile: 'Dockerfile_jenkins') {
      withEnv([
        "SRC=$WORKSPACE",
        "BUILD=$WORKSPACE/build"
      ]) {
        stage('release') {
          sh 'mkdir $BUILD'
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
        stage('asan_and_ubsan') {
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
  },
  cuda: {
    buildPod(context: 'docker', dockerfile: 'Dockerfile_jenkins_cuda', tag: 'cuda', gpus: 1) {
      withEnv([
        "SRC=$WORKSPACE",
        "BUILD=$WORKSPACE/build"
      ]) {
        stage('cuda') {
          sh 'mkdir $BUILD'
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
