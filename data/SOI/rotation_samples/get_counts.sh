#!/usr/bin/env bash

for i in *.eul;

  do

    wc -l $i >> gridpoints_count.txt;

done;
