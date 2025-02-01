for dir in baseline/*; do
                       if [ -d "$dir/commits" ]; then
                           git checkout HEAD -- "$dir/commits/*"
                             fi
                             done

