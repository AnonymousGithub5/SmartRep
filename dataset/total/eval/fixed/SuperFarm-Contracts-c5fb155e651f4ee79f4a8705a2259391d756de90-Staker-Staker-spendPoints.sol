function FunctionDefinition_0 ( address _user , uint256 _amount ) external { require ( Identifier_0 [ msg . sender ] , stringLiteral_0 ) ; uint256 VariableDeclaration_0 = Identifier_1 ( _user ) ; require ( Identifier_2 >= _amount , stringLiteral_1 ) ; Identifier_3 [ _user ] = Identifier_4 [ _user ] + _amount ; emit Identifier_5 ( msg . sender , _user , _amount ) ; }