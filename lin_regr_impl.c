#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#define NUM_FEATURES 3
#define NUM_SAMPLES 40

void linear_regression_train(float X_train[][NUM_FEATURES],float *y_train, int num_samples, int num_features);
void  print_weights(float *weights, int num_weights);
float mse_loss(float *weights, float X_train[][NUM_FEATURES], float *y_train, int num_samples, int num_features);
void compute_gradients(float *weights, float *gradients, float X_train[][NUM_FEATURES], float *y_train, int num_samples, int num_features);

void update_weights(float *weights, float *gradients, float lr, int num_features);
void linear_regression_predict(float *weights, float X_train[][NUM_FEATURES], int num_samples, int num_features, float *predictions);
void save_weights(float *weights, int num_weights, char *filename);
void load_weights(float *weights, int num_weights, char *filename);


void save_weights(float *weights, int num_weights, char *filename)
{
    /*This will save the weights in a file*/
    /*We need to persist this in a table eventually*/
    FILE *fp = NULL;

    printf("Saving weights\n");

    fp=fopen(filename, "wb");
    //Please serialize the weights array into the file pointed by fp
    fclose(fp);
}

void load_weights(float *weights, int num_weights, char *filename)
{
    /*Make sure you allocate memory 1+num_features to consider the bias also*/
    /*This function should be modified so that it reads from the table instead of a binary file*/
    FILE *fp = NULL;
    printf("Loading weights\n");
    
    fp=fopen(filename, "rb");
    //Add code to read weights from the file using fp
    fclose(fp);
}
    
    
void update_weights(float *weights, float *gradients, float learning_rate, int num_features)
{

    //Update the current set of weights using the passed gradients and the given learning rate
    //Remember that the number of elements in gradients is equal to the number of elements in the weights. Why?
    //You can use the same weights as input and output params, you compute the new weights and you overwrite the existing weights variable
    //The code should look like
    //weights[i] = expression involving weights[i], gradients[i] and the learning rate

}
void compute_gradients(float *weights, float *gradients, float X_train[][NUM_FEATURES], float *y_train, int num_samples, int num_features)
{

    //Compute the gradient of the loss function with the given training data and targets and given set of weights
    //Populate the gradients variable which is an output parameter
    //Remember that the length of the gradients is the same as that of the weights, Why?
    //WRITE CODE FOR COMPUTING THE GRADIENT
}

void linear_regression_predict(float *weights, float X_train[][NUM_FEATURES], int num_samples, int num_features, float *preds)
{

    //X_train is the input
    //Please predict using the weights passed and update the predictions in the variable preds
    //Remember preds is a 1-d array of float of size num_samples,
    //as there is 1 prediction to be done for each sample
}


float mse_loss(float *weights, float X_train[][NUM_FEATURES], float *y_train, int num_samples, int num_features)
{
    //Implement the mean square error loss function here
    //This function should return the average loss of the entire dataset using the given set of weights
}


void  print_weights(float *weights, int num_weights)
{
    int i;
    printf("Coefficients:[  ");
    for(i=0;i<num_weights-1;i++)
    {
        printf("%f  ", weights[i]);
    }
    printf("]  ");
    printf("Intercept/Bias:%f\n", weights[num_weights-1]);
    //Assume that the last weight is the intercept/bias
}


void linear_regression_train(float X_train[][NUM_FEATURES],float *y_train, int num_samples, int num_features)
{
    int i, j;
    int num_weights=num_features + 1;
    int epoch=0;
    float weights[NUM_FEATURES+1]={0}; //Why should we add +1 here?
    float gradients[NUM_FEATURES];
    float predictions[NUM_SAMPLES]; //Why is the length of this array NUM_SAMPLES?
    float loss = 0.0f;
    float new_loss = 0.0f;
    float learning_rate=0.01;
    float epsilon = 1e-6;
    float delta;
    int num_epochs = 10;
    /*Print the dataset to check everything fine*/
    for(i = 0;i < num_samples; i++)
    {
        for(j = 0; j < num_features; j++)
            printf("%f\t", X_train[i][j]);
        printf("Label=%f", y_train[i]);
        printf("\n");
    }

    print_weights(weights, num_weights);
    //Compute loss of the entire training set. Please implement the mse_loss function
    loss = mse_loss(weights, X_train, y_train, num_samples, num_features);
    printf("Initial Loss=%f\n", loss);
    for(epoch=0;epoch<num_epochs;epoch++)
    {
        printf("Running epoch %d with learning_rate=%f\n", 1+epoch, learning_rate);
        //STEP 1:
        //Compute the gradient of the entire dataset.
        //You can create a compute_gradients function to calculate the gradients
        //Prototype is already given. Feel free to add/remove extra/unnecessary params if required
        //Remember that the gradient is a vector of size num_features+1
        //WRITE CODE HERE to call the compute_gradients function whose prototype is already given

        //STEP 2:
        //Update the weights using the learning rate and the computed gradient
        //Remember that the weights are also a vector of size num_features+1
        //WRITE CODE FOR STEP 2
        
        //STEP3:
        //Compute the new loss with the updated weights
        //Ensure that the loss has decreased
        //If the difference between the previous loss and the current loss is less than epsilon
        //you can exit the training by calling a break
        //WRITE CODE FOR STEP 3


    }
    print_weights(weights, num_weights);
    //Should ideally print 3, 4, -5 and 7 as intercept because we used that formula
    save_weights(weights, num_weights, "lr_model.bin");

    //Initialize all weights to 0
    memset(weights, 0, num_weights*sizeof(float));

    //Load weights by calling load_weights
    load_weights(weights, num_weights, "lr_model.bin");
    printf("Weights after loading from the model\n");
    print_weights(weights, num_weights); //Should print the original weights saved
    

}
/*A simple demo to test linear_regression_training*/
int main(int argc, char *argv[])
{
    float X_train[NUM_SAMPLES][NUM_FEATURES]={0};
    float y_train[NUM_SAMPLES]={0};
    int i=0, j=0;

    //Lets generate some synthetic data using a simple linear expression
    //Lets use y = 3*x1 + 4*x2 - 5*x3 + 7 to generate NUM_SAMPLES
    //Idea is to learn the coefficients 3, 4, -5 and 7 from the given data

    for(i=0;i<NUM_SAMPLES;i++)
    {
        //Generate some random numbers as x1, x2 and x3
        X_train[i][0] = (float)((i+10)%7 );
        X_train[i][1] = (float)((i+13) % 9);
        X_train[i][2] = (float)((i-5) % 10);

        //Use the formula to generate its y

        y_train[i]= 3*X_train[i][0] +4*X_train[i][1] -5*X_train[i][2] + 7 ;
    }

    //Call the linear regression train function with X_train, y_train
    //PLEASE FOLLOW THE INSTRUCTIONS in the linear_regression_train to fill the missing/required code

    linear_regression_train(X_train, y_train, NUM_SAMPLES, NUM_FEATURES);
}


//Some questions to ponder
//We initialized the weights to 0 in the linear_regression_train function
//Why did we initialize to 0, what would happen if we initialized to random numbers?
//Please try initializing to random numbers around 0 to 1 and see what happens to the weights eventually
//What happens if you use a very big learning_rate?
//What happens if you use a very small learning_rate?
//How can you ensure that the algorithm converges faster?
//In this example, we hard coded all matrices using a #define, like NUM_FEATURES and NUM_SAMPLES.
//In real world, we don't know how many features in advance, our algorithm should be able to work with any number of features - So how can you do dynamic passing of arrays whose size is determined during runtime and not during compile time?
//If you find any thing ambiguous, please make sensible assumptions and proceed, and mention those assumptions and the rationale behind it.
