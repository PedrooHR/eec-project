#!/usr/bin/env bash

# define 
calc() { awk "BEGIN{print $*}"; }

# get each applications
applications=$(ls -d */) 
for app in $applications; do 
    #get each scenario
    scenarios=$(ls -d $app*/)
    for sce in $scenarios; do
        echo "" > ${sce}sorgan1-report.txt
        echo "" > ${sce}sorgan2-report.txt
        echo "" > ${sce}sorgan3-report.txt
        #get each sample
        samples=$(ls -d $sce*/)
        for spl in $samples; do
            echo $spl
            sorgan1=$(ls ${spl}sorgan-gpu1*)
            sorgan2=$(ls ${spl}sorgan-gpu2*)
            sorgan3=$(ls ${spl}sorgan-gpu3*)

            #for sorgan 1
            count=0;
            pkg_sum=0;
            ram_sum=0;
            time_sum=0;
            for file in $sorgan1; do
                pkg=$(cat $file | grep 'pkg') && pkg=($pkg) && pkg=${pkg[0]} && pkg_sum=$(calc $pkg_sum + $pkg)
                ram=$(cat $file | grep 'ram') && ram=($ram) && ram=${ram[0]} && ram_sum=$(calc $ram_sum + $ram)
                time=$(cat $file | grep 'time') && time=($time) && time=${time[0]} && time_sum=$(calc $time_sum + $time)
            done
            pkg_sum=$(calc $pkg_sum / 10.0)
            ram_sum=$(calc $ram_sum / 10.0)
            time_sum=$(calc $time_sum / 10.0)
            echo $pkg_sum  >> ${sce}sorgan1-report.txt
            echo $ram_sum  >> ${sce}sorgan1-report.txt
            echo $time_sum >> ${sce}sorgan1-report.txt

            #for sorgan 2
            count=0;
            pkg_sum=0;
            ram_sum=0;
            time_sum=0;
            for file in $sorgan2; do
                pkg=$(cat $file | grep 'pkg') && pkg=($pkg) && pkg=${pkg[0]} && pkg_sum=$(calc $pkg_sum + $pkg)
                ram=$(cat $file | grep 'ram') && ram=($ram) && ram=${ram[0]} && ram_sum=$(calc $ram_sum + $ram)
                time=$(cat $file | grep 'time') && time=($time) && time=${time[0]} && time_sum=$(calc $time_sum + $time)
            done
            pkg_sum=$(calc $pkg_sum / 10.0)
            ram_sum=$(calc $ram_sum / 10.0)
            time_sum=$(calc $time_sum / 10.0)
            echo $pkg_sum  >> ${sce}sorgan2-report.txt
            echo $ram_sum  >> ${sce}sorgan2-report.txt
            echo $time_sum >> ${sce}sorgan2-report.txt

            #for sorgan 3
            count=0;
            pkg_sum=0;
            ram_sum=0;
            time_sum=0;
            for file in $sorgan3; do
                pkg=$(cat $file | grep 'pkg') && pkg=($pkg) && pkg=${pkg[0]} && pkg_sum=$(calc $pkg_sum + $pkg)
                ram=$(cat $file | grep 'ram') && ram=($ram) && ram=${ram[0]} && ram_sum=$(calc $ram_sum + $ram)
                time=$(cat $file | grep 'time') && time=($time) && time=${time[0]} && time_sum=$(calc $time_sum + $time)
            done
            pkg_sum=$(calc $pkg_sum / 10.0)
            ram_sum=$(calc $ram_sum / 10.0)
            time_sum=$(calc $time_sum / 10.0)
            echo $pkg_sum  >> ${sce}sorgan3-report.txt
            echo $ram_sum  >> ${sce}sorgan3-report.txt
            echo $time_sum >> ${sce}sorgan3-report.txt
        done
    done
done

# sorgan1=$(ls sorgan-gpu1*)
# sorgan2=$(ls sorgan-gpu2*)
# sorgan3=$(ls sorgan-gpu3*)

# for f in sorgan1; do
#     pkg=$(cat $f | grep 'pkg') && pkg=($pkg) && pkg=${pkg[0]};
#     ram=$(cat $f | grep 'ram') && ram=($ram) && ram=${ram[0]};
#     time=$(cat $f | grep 'time') && time=($time) && time=${time[0]};
# done