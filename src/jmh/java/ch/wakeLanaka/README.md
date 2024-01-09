# Benchmarks

Unfortunately, does gradle currently not support Java version 21. Therefore, to
execute these benchmark the following steps are needed:

1. Open a terminal where JAVA_HOME has a version supported by gradle
2. execute `gradle build`
3. Open a new terminal where JAVA_HOME has the path to the binaries to the
   OpenJDK JDK with SVMBuffer API support. (panama-vector/build/linux-x86_64-server-release/jdk/)
4. execute `gradle jmh` with the terminal where JAVA_HOME has SVMBuffer API
   support.
5. Every time build.gradle is changes the steps from 1 to 4 have to be repeated
