function FunctionDefinition_0 ( address Parameter_0 , uint256 Parameter_1 , uint256 calldata Parameter_2 ) external onlyAdmin { IERC20 token = IERC20 ( tokenAddress ) ; uint256 VariableDeclaration_0 = token . balanceOf ( address ( this ) ) ; token . safeApprove ( address ( Identifier_0 ) , Identifier_1 ) ; uint256 VariableDeclaration_1 = Identifier_2 . balanceOf ( address ( this ) ) ; Identifier_3 . MemberAccess_0 ( tokenAddress , address ( Identifier_4 ) , Identifier_5 , Identifier_6 , Identifier_7 , 0 ) ; uint256 VariableDeclaration_2 = Identifier_8 . balanceOf ( address ( this ) ) ; require ( Identifier_9 > Identifier_10 , stringLiteral_0 ) ; }