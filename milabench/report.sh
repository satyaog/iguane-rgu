#!/bin/bash

_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)

function to_csv {
    _header=1

    while read d
    do
        grep -E '"return_code": 0|Terminating process because it ran for longer than|early_stop' "$d"*.data >/dev/null
        if [[ "$?" != "0" ]]
        then
            1>&2 echo SKIPPING "$d"
            continue
        fi

        _dir=$(basename "$d")
        gpu=$(echo "$_dir" | cut -d"_" -f1)

        _i=3
        while [[ ! "$(echo "$_dir" | cut -d"_" -f"$_i" | cut -d"." -f1)" -gt "0" ]]
        do
            1>&2 echo "$_dir" | cut -d"_" -f"$_i" | cut -d"." -f1

            if [[ -z "$(echo "$_dir" | cut -d"_" -f"$_i" | cut -d"." -f1)" ]]
            then
                break
            fi
            _i=$(($_i + 1))
        done

        bench=$(echo "$_dir" | cut -d"_" -f2-$(($_i - 1)))
        batch_size=$(echo "$_dir" | cut -d"_" -f$_i | cut -d"." -f1)
        1>&2 echo "$d"
        1>&2 echo "$gpu"
        1>&2 echo "$bench"
        1>&2 echo "$batch_size"

        if [[ -z "$batch_size" ]]
        then
            1>&2 echo SKIPPING "$d"
            continue
        fi

        while read l
        do
            1>&2 echo "$l"

            bench=$(echo $l | cut -d"|" -f1)
            if [[ "$bench" == "bench"* ]]
            then
                if [[ -z "${_header}" ]]
                then
                    continue
                fi 
                l=$(echo "gpu" "|" $bench "|" "batch_size" "|" $(echo $l | cut -d"|" -f2-))
                unset _header
            else
                l=$(echo $gpu "|" $bench "|" $(printf "%09d" $batch_size) "|" $(echo $l | cut -d"|" -f2-))
            fi
            l=${l// | /,}
            1>&2 echo "$l"

            echo "$l"
        done < <(hatch run milabench report --runs "$d" | grep " | ")
    done
}

echo $_HERE

to_csv 2>"${_HERE}"/report.err >"${_HERE}"/report.csv < <(ls -d "${_HERE}"/runs/*/ | grep -Ev "failed|staging")

to_csv 2>"${_HERE}"/report.failed.err >"${_HERE}"/report.failed.csv < <(ls -d "${_HERE}"/runs/*/ | grep -E "failed|staging")
