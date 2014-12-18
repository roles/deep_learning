#ifndef _CONFIG_H
#define _CONFIG_H

const int maxLayer = 5;
const int maxUnit = 2005;
const int maxBatchSize = 10;

const int numBatchPerLog = 0;

const double L2Reg = 0.00001;

typedef double RBMBuffer[maxUnit*maxBatchSize];


#endif
