#ifndef _CONFIG_H
#define _CONFIG_H

const int maxLayer = 5;
const int maxUnit = 5005;
const int maxBatchSize = 3;

const int numBatchPerLog = 0;

const double L2Reg = 0.00001;

typedef double RBMBuffer[maxUnit*maxBatchSize];


#endif
