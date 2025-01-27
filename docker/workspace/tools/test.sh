#!/bin/bash
analyzer_jar=/root/workspace/tools/java_analyzer/target/java-analyzer-1.0-SNAPSHOT-shaded.jar
#file=/root/workspace/docker/workspace/tools/copy.java
file=copy.java

# use Java 11
source $SDKMAN_DIR/bin/sdkman-init.sh && sdk use java 11.0.12-open

java -cp $analyzer_jar analyzer.RewriteRunner $file cleanup.diff
patch -p1 < cleanup.diff
rm cleanup.diff