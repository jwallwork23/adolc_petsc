#define MAXSIZE 100
int AD_PP_STACK[MAXSIZE];
int AD_PP_TOP = 0;

void pushControlNb(int val) {
    AD_PP_STACK[AD_PP_TOP++] = val;
}

void pushControl1b(int val) {
    pushControlNb(val);
}

void pushControl2b(int val) {
    pushControlNb(val);
}

void pushControl3b(int val) {
    pushControlNb(val);
}



void popControlNb(int *val) {
    *val = AD_PP_STACK[--AD_PP_TOP];
}

void popControl1b(int *val) {
    popControlNb(val);
}

void popControl2b(int *val) {
    popControlNb(val);
}

void popControl3b(int *val) {
    popControlNb(val);
}
