function FunctionDefinition_0 ( address _token , uint amount ) virtual external override onlyOwner { require ( address ( Identifier_0 ) != _token , stringLiteral_0 ) ; Identifier_1 ( _token ) . safeTransfer ( owner ( ) , amount ) ; emit Identifier_2 ( _token , amount ) ; }