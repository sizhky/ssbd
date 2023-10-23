# for row_index in 1 3 6 8 11 13 15 18 20 22 25 28 30 32 35 37 39 41 43 45 47 49 51 53 55 57 59 61 63 65 67 69 71 73 75 77 79 81 83 85 87 89 91 93 95 97 99 101 103 105 107 109 111 113 115 117 119 121 123 125 127 129 131 133 136 138 140 142 145 147 149 151 154 156 158 160 163 166 169 172 174 177 179 182 185 187 190 192 195 198 200 203 205 208 210 213 215 218 220 223 225 228 231 233 236 238 241 243 246 248 251 254 257 260 262 265 267 270 273 275 278 280 283 286 289 292 295 298 301 304 307 310 313 315 317 319 321 323 325 327 329 331 333 336 338 340 343 345 347 350 352 355 357 360 362 365 368 371 374 378 381 384 387 389 392 394 397 399 402 405 409 412 415 418 421 423 425 427 430 432 434 436 439 441 443 445 448 450 452 454 457 459 461 463 466 469 472 475 479
for row_index in $(seq 0 481)
do
    echo $row_index
    clip-video-classifier extract-frames-for-annotations \
        # annotations csv
        /mnt/347832F37832B388/ml-datasets/ssbd/annotations.csv \ 
        # raw videos path
        /mnt/347832F37832B388/ml-datasets/ssbd/ssbd-raw-videos/ \
        # path to store annotations and number of frames per second, 
        # higher the frames, better the data quality and larger the storage volue
        /mnt/347832F37832B388/ml-datasets/ssbd/ssbd-frames/10fps/ --num-frames-per-sec 10 \
        --no-exclude-others \
        --row-index=$row_index &
    # Limit the number of parallel processes to as many as you want, default 1
    if [ $(jobs | wc -l) -ge 0 ]; then
        wait
    fi
done

# Ensure all background jobs are finished
wait
