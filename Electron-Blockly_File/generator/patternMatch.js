// Assuming the JSON is stored in a variable called `jsonData`
const jsonData = {
    "CNN": {
        "functions": [
            {
                "functionName": "__call__",
                "parameters": "(self,x)",
                "functionCalls": [
                    {
                        "function": "Conv",
                        "parameters": "(features=32, kernel_size=(3, 3))(x)"
                    },
                    {
                        "function": "Conv",
                        "parameters": "(features=32, kernel_size=(3, 3))"
                    }
                ],
                "scope": "CNN",
                "row": "3",
                "start_index": 38,
                "translate": true
            }
        ],
        "translate": true,
        "row": "1",
        "start_index": 0,
        "parameters": "",
        "base_classes": "(nn.Module)"
    },
    "global": {
        "functions": [
            {
                "functionName": "update_model",
                "parameters": "(state, grads)",
                "functionCalls": [
                    {
                        "function": "Conv",
                        "parameters": "(features=32, kernel_size=(3, 3))(x)"
                    },
                    {
                        "function": "Conv",
                        "parameters": "(features=32, kernel_size=(3, 3))"
                    },
                    {
                        "function": "apply_gradients",
                        "parameters": "(grads=grads,foo())"
                    },
                    {
                        "function": "foo",
                        "parameters": "()"
                    }
                ],
                "scope": "",
                "row": "8",
                "start_index": 117,
                "translate": false
            },
            {
                "functionName": "foo",
                "parameters": "(array)",
                "functionCalls": [
                    {
                        "function": "Conv",
                        "parameters": "(features=32, kernel_size=(3, 3))(x)"
                    },
                    {
                        "function": "Conv",
                        "parameters": "(features=32, kernel_size=(3, 3))"
                    }
                ],
                "scope": "",
                "row": "14",
                "start_index": 256,
                "translate": true
            }
        ],
        "translate": false
    }
};

// Iterate over the top-level keys in the JSON
Object.keys(jsonData).forEach(key => {
    const object = jsonData[key];

    // Check if translate is true at the object level
    if (object.translate) {
        console.log(`Processing object at the top level: ${key}`);

        // Iterate through the functions if they exist
        if (object.functions) {
            object.functions.forEach(func => {
                // If the function-level translate flag is true
                if (func.translate) {
                    console.log(`Function to translate: ${func.functionName}`);

                    // Iterate through functionCalls
                    func.functionCalls.forEach(call => {
                        console.log(`Processing function call: ${call.function} with parameters ${call.parameters}`);
                        // Do something with each function call
                    });
                }
            });
        }
    }

    // Iterate through the functions if they exist
    if (object.functions) {
        object.functions.forEach(func => {
            // If the function-level translate flag is true
            if (func.translate) {
                console.log(`Function to translate: ${func.functionName}`);

                // Iterate through functionCalls
                func.functionCalls.forEach(call => {
                    console.log(`Processing function call: ${call.function} with parameters ${call.parameters}`);
                    // Do something with each function call
                });
            }
        });
    }
});
