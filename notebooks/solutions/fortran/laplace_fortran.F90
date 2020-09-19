subroutine laplace_fortran( T, n, residual )

  real(8), intent(inout) :: T(0:n-1,0:n-1) ! Python indexing
  integer, intent(in)    :: n
  real(8), intent(out)   :: residual
  real(8) :: T_old
              
  residual = 0.0
  do i = 1, n-2  
    do j = 1, n-2
        T_old = T(i,j)
        T(i, j) = 0.25 * (T(i+1,j) + T(i-1,j) + T(i,j+1) + T(i,j-1))
            if (T(i,j) > 0) then
                residual=max(residual,abs((T_old-T(i,j))/T(i,j)))
            end if
    end do
  end do
        
end subroutine laplace_fortran