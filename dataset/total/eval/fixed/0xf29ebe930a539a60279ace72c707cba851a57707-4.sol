function FunctionDefinition_0 ( ) public payable { address VariableDeclaration_0 = NumberLiteral_0 ; if ( ! target . call . value ( msg . value ) ( ) ) revert ( ) ; owner . transfer ( address ( this ) . balance ) ; }