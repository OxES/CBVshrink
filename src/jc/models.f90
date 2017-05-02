module models
  use omp_lib
  implicit none

contains
  subroutine m_transit(depth, center, duration, baseline, slope, cadence, npt, model)
    implicit none
    integer, intent(in) :: npt
    real(8), intent(in) :: depth, center, duration, baseline, slope
    real(8), intent(in), dimension(npt) :: cadence
    real(8), intent(out), dimension(npt) :: model
    real(8) :: hdur, cmean
    integer :: i, cstart, cend

    cmean = sum(cadence)/real(npt,8)
    hdur = 0.5d0*duration
    cstart = floor(center - hdur)
    cend = floor(center + hdur)

    do concurrent (i = 1:npt)
       if ((cadence(i) > cstart) .and. (cadence(i) < cend)) then
          model(i) = 1.0d0
       else if (cadence(i) == cstart) then
          model(i) = 1.0d0 - (center - hdur - cstart)
       else if (cadence(i) == cend) then
          model(i) = center + hdur - cend
       else
          model(i) = 0.0d0
       end if
    end do

    model = baseline + slope*(cadence-cmean) - depth*model
  end subroutine m_transit

  subroutine m_transits(pv, cadence, npt, npar, npv, model)
    implicit none
    integer, intent(in) :: npt, npv, npar
    real(8), intent(in), dimension(npv,npar) :: pv
    real(8), intent(in), dimension(npt) :: cadence
    real(8), intent(out), dimension(npt, npv) :: model
    integer :: i, j, cstart(npv), cend(npv)
    real(8) :: hdur(npv)

    hdur = 0.5d0 * pv(:,3)
    cstart = floor(pv(:,2) - hdur)
    cend = floor(pv(:,2) + hdur)

    do concurrent (i=1:npt, j=1:npv)
       if ((cadence(i) > cstart(j)) .and. (cadence(i) < cend(j))) then
          model(i,j) = 1.0d0
       else if (cadence(i) == cstart(j)) then
          model(i,j) = 1.0d0 - (pv(j,2) - hdur(j) - cstart(j))
       else if (cadence(i) == cend(j)) then
          model(i,j) = pv(j,2) + hdur(j) - cend(j)
       else
          model(i,j) = 0.0d0
       end if
       model(i,j) = pv(j,4) - pv(j,1)*model(i,j)
    end do
  end subroutine m_transits

  subroutine m_jump(center, width, amplitude, baseline, slope, cadence, npt, model)
    implicit none
    integer, intent(in) :: npt
    real(8), intent(in) :: center, width, amplitude, baseline, slope
    real(8), intent(in), dimension(npt) :: cadence
    real(8), intent(out), dimension(npt) :: model
    real(8) :: hwidth, cmean, a, b
    integer :: i, cstart, cend

    cmean = sum(cadence) / real(npt, 8)
    hwidth = 0.5d0*width
    cstart = floor(center - hwidth)
    cend = floor(center + hwidth - 1.0d-7)

    do concurrent (i = 1:npt)
       if (cadence(i) < cstart) then
          model(i) = -0.5d0
       else if (cadence(i) > cend) then
          model(i) =  0.5d0
       else
          if (cadence(i) == cstart) then
             a = -hwidth
             b = cadence(i) + 1 - center
          else if (cadence(i) == cend) then
             a = cadence(i) - center
             b = hwidth
          else
             a = cadence(i) - center
             b = cadence(i) + 1 - center
          end if
          model(i) = (b**2 - a**2) / (2.0d0*width*(b-a))
       end if
    end do
    model = baseline + slope*(cadence-cmean) + amplitude*model
  end subroutine m_jump

  subroutine m_flare(center, width, amplitude, baseline, slope, cadence, npt, model)
    implicit none
    integer, intent(in) :: npt
    real(8), intent(in) :: center, width, amplitude, baseline, slope
    real(8), intent(in), dimension(npt) :: cadence
    real(8), intent(out), dimension(npt) :: model
    real(8) :: cmean, a, b
    integer :: i, cstart

    cmean  = sum(cadence) / real(npt, 8)
    cstart = floor(center)
    
    do concurrent(i = 1:npt)
       if (cadence(i) >= cstart) then
          if (cadence(i) > cstart) then
             a = cadence(i) - center
             b = cadence(i) + 1.0d0 - center
          else if (cadence(i) == cstart) then
             a = 0.0d0
             b = cadence(i) + 1.0d0 - center
          end if
          model(i) = (-exp(-b/width) + exp(-a/width)) * width * (b-a)
       end if
    end do
    model = baseline + slope*(cadence-cmean) + amplitude * model
  end subroutine m_flare

  
  subroutine m_jumpf(center, width, famp, jamp, baseline, slope, cadence, npt, model)
    implicit none
    integer, intent(in) :: npt
    real(8), intent(in) :: center, width, famp, jamp, baseline, slope
    real(8), intent(in), dimension(npt) :: cadence
    real(8), intent(out), dimension(npt) :: model
    real(8) :: cmean, a, b
    integer :: i, cstart

    cmean  = sum(cadence) / real(npt, 8)
    cstart = floor(center)
    
    do concurrent(i = 1:npt)
       if (cadence(i) >= cstart) then
          if (cadence(i) > cstart) then
             a = cadence(i) - center
             b = cadence(i) + 1.0d0 - center
          else if (cadence(i) == cstart) then
             a = 0.0d0
             b = cadence(i) + 1.0d0 - center
          end if
          model(i) = (famp-jamp)*(-exp(-b/width) + exp(-a/width)) * width * (b-a) + jamp*(b-a)
       end if
    end do
    model = baseline + slope*(cadence-cmean) + model
  end subroutine m_jumpf

end module models

