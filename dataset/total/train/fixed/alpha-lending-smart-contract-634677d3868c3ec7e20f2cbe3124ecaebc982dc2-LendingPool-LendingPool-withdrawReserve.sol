function FunctionDefinition_0 ( ERC20 _token , uint256 _amount ) external nonReentrant ModifierInvocation_0 ( _token ) onlyOwner { UserDefinedTypeName_0 storage pool = Identifier_0 [ address ( _token ) ] ; uint256 VariableDeclaration_0 = _token . balanceOf ( address ( this ) ) ; require ( _amount <= Identifier_1 , stringLiteral_0 ) ; require ( _amount <= pool . MemberAccess_0 , stringLiteral_1 ) ; _token . safeTransfer ( msg . sender , _amount ) ; pool . MemberAccess_1 = pool . MemberAccess_2 . sub ( _amount ) ; emit Identifier_2 ( address ( _token ) , _amount , msg . sender ) ; }