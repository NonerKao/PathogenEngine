#!/usr/bin/awk -f

BEGIN {
    FS = ",";
}

{
    sum1 += $1;
    sum2 += $2;
    sum3 += $3;
    sum4 += $4;
}

END {
    if (sum4 == 0) {
        print "Sum of the fourth column is zero, cannot divide by zero.";
        exit 1;
    }
    print "Stay rate: " (sum1/sum4)*100 "%";
    print "Bad policy rate: " (sum2/sum4)*100 "%";
    print "Invalidity rate: " (sum3/sum4)*100 "%";
}

