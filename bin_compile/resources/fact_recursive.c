// Function to calculate factorial
unsigned long long factorial(int n) {
    if (n < 0) {
        return 0; // Factorial is not defined for negative numbers
    }
    if (n == 0 || n == 1) {
        return 1; // Base case: 0! = 1! = 1
    }
    return n * factorial(n - 1); // Recursive case
}
