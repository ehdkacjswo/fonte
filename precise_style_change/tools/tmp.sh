#!/bin/bash
analyzer_jar=/root/workspace/docker/workspace/tools/java_analyzer/target/java-analyzer-1.0-SNAPSHOT-shaded.jar
# use Java 11
source $SDKMAN_DIR/bin/sdkman-init.sh && sdk use java 11.0.12-open

after_src_path=/root/workspace/data_collector/copy.java

java -cp $analyzer_jar analyzer.RewriteRunner $after_src_path cleanup.diff
patch -p1 < cleanup.diff
rm cleanup.diff