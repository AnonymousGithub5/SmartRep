function withdraw ( uint256 _amount ) external { UserInfo storage user = userInfo [ msg . sender ] ; require ( user . amount >= _amount , stringLiteral_0 ) ; Identifier_0 ( ) ; uint256 VariableDeclaration_0 = user . amount * Identifier_1 / 1e12 - user . rewardDebt ; if ( pending > 0 ) { user . MemberAccess_0 = user . MemberAccess_1 + pending ; } if ( _amount > 0 ) { user . amount = user . amount - _amount ; Identifier_2 = Identifier_3 - _amount ; user . MemberAccess_2 = user . MemberAccess_3 + _amount ; } user . rewardDebt = user . amount * Identifier_4 / 1e12 ; emit Identifier_5 ( msg . sender , _amount ) ; }